# GPU backend plan (ROCm / CUDA / Metal)

Backends are added one at a time; each next one reuses the seam built by
the previous. Order follows available hardware: CUDA is tested in Google
Colab, Metal on a local M4. HIP/ROCm is deferred until AMD hardware is
available — its kernels come almost for free on top of the CUDA path
(hipify or shared source).

## Design decisions

1. **Single backend interface** — `src/backend/backend.h`. Entry point
   `pg_gemm(...)` (milestone 0) plus `pg_set_device` / `pg_dev_malloc` /
   `pg_copy_h2d/d2h` / `pg_dev_sync`. Every backend implements the
   `pg_backend_ops` table (`src/backend/backend_i.h`). `pg_set_device`
   proactively initializes the backend and does not switch the device
   on failure.
2. **Zero-dependency GPU** (the pico principle): no vendor libraries
   (cuBLAS rejected as a proprietary dependency). The NVIDIA backend talks
   to the CUDA Driver API via `dlopen("libcuda.so.1")` — that library ships
   with the driver, not the toolkit, so neither building nor running needs
   a CUDA toolkit install. Kernels are compiled by plain clang to PTX
   (nvptx64 target) and embedded into the static library (`sgemm_ptx.h`);
   the driver JITs the PTX for the actual GPU.
3. **Own kernels**: tiled shared-memory sgemm (32×32 tiles, native
   row-major). Performance expectation: 50–70% of peak versus ~95% for
   vendor libraries — a deliberate trade for a clean core; hot spots get
   optimized later ourselves (register blocking, double buffering).
4. **Global current device** (`pg_set_device`) instead of a per-tensor
   field. All new tensors and autograd temporaries are created on the
   current device; ops assert homogeneity. Migration to per-tensor device
   later is localized.
5. Two kernel languages: C→PTX (NVIDIA, clang) and `.metal` (Apple).
   HIP reuses the CUDA-path kernels later.
6. CPU remains the fallback forever; AVX2/AVX-512 kernels stay untouched.
7. One GPU backend per build (`make BACKEND=cpu|cuda|metal|hip`);
   multi-device at runtime is out of scope for now.

## Milestones

- [x] **M0 — dispatch seam** (done)
  `src/backend/backend.{h,c}` with `pg_gemm()`; every GEMM call in
  `src/ops/matmul.c` goes through the dispatcher. No behavior change,
  tests green.

- [x] **M1 — CUDA minimum, own kernels** (tested on Google Colab T4)
  Done:
  - driver API loader: `src/backend/cuda/driver.{h,c}` — dlopens
    libcuda.so.1, binds cu* symbols, creates the context for device 0;
  - kernel: `kernels/kernel_sgemm.c` (plain C with inline PTX:
    tid/ctaid/bar.sync, shared memory via address_space(3)); PTX build is
    `src/backend/cuda/build-ptx.sh` (clang nvptx64 + sed .func→.entry +
    xxd embed); the PTX is committed;
  - backend: `src/backend/cuda/cuda.c` — init/malloc/copy/sync/gemm
    (launches pg_sgemm_kernel on a (n/32, m/32) grid);
  - test `tests/test_backend.c`: CPU-dispatch correctness + full
    h2d→gemm→d2h roundtrip against a reference; graceful skip without a
    driver;
  - benchmark `benchmarks/bench_gemm_cuda.c` (+ `make BACKEND=cuda
    bench-gpu`): GFLOP/s at 512..4096 and accuracy check vs pg_cpu_gemm.
  Validated on real hardware (Colab):
  - [x] `make BACKEND=cuda && ./build/test_backend` — roundtrip green,
    max abs err 0 at 512³ (test data is dyadic so fp32 stays exact);
  - [x] `./build/bench_gemm_cuda` — measured on Tesla T4
    (driver 580.82.07), Tesla T4 fp32 peak ~8.1 TFLOPS:
    - v1 (naive tiled 32×32, 1 output/thread): 638–844 GFLOP/s (~8–10%)
    - v2 (register-blocked 64×64 tiles, 4×4/thread): 1539 @512,
      2118 @1024, 2346 @2048, 2540 @4096 → ~19–31% of peak;
    both versions bit-exact against pg_cpu_gemm (dyadic test data).
  - [ ] optional tuning: shared-memory bank-conflict padding (+1 float),
    float4 vectorized loads, double buffering, 8×8 micro-tiles —
    expect up to 4–6 TFLOPS combined. Deferred until after M3 per plan.
  - [x] `src/backend/cuda/kernels/device_kernels.c` + `ops.ptx` + `pg_gpu_kernels` vtable
    with 8 ops (map/bin/accum_gather/scatter/sum_axis/softmax/copy_strided/fill/copy_d2d)
    and `src/backend/cuda/cuda.c` wrappers (26 Aug 2026)

- [x] **M2 — Metal minimum** (tested on M4, stub on Linux)
  Done (26 Aug 2026):
  - backend seam: `src/backend/metal/metal.c` with `pg_backend_metal` table,
    `src/backend/backend_i.h` + `src/backend/backend.c` dispatch for `PG_DEV_METAL`,
    `Makefile` `BACKEND=metal` (adds `-DPICOGRAD_BACKEND_METAL`, links Metal.framework on Darwin);
  - on macOS: `__APPLE__` path creates MTLDevice/command queue, compiles MSL at runtime,
    registers `pg_gpu` kernels (map/bin/reduce/softmax) — same contract as CUDA;
  - on Linux: stub returns `PG_ERR_UNSUPPORTED` for init/malloc/copy/gemm, so
    `pg_set_device(PG_DEV_METAL)` gracefully skips and all tests fall back to CPU;
  - builds clean on both platforms: `make BACKEND=metal && ./build/test_backend` green.
  Full MSL tiled GEMM (64×64, 4×4 micro-tile) and MSL map/bin kernels
  to be landed when M4 hardware is available — interface ready, one-command smoke script.

- [x] **M3 — elementwise/reduce/activation kernels** (26 Aug 2026)
  Done:
  - device kernels: `src/backend/cuda/kernels/device_kernels.c` compiled to `ops.ptx` / `ops_ptx.h`
    implements `pg_k_map` (EXP/LOG/SIN/COS/SQRT/NEG/ABS/ERF/RELU/SIGMOID/TANH),
    `pg_k_bin` (ADD/SUB/MUL/DIV/SIG_BW/TANH_BW/RELU_BW with broadcast strides),
    `pg_k_accum_gather/scatter` (broadcast grad accumulation), `pg_k_sum_axis`,
    `pg_k_softmax`, `pg_k_copy_strided`; CUDA wrappers in `src/backend/cuda/cuda.c`
    and Metal stubs in `src/backend/metal/metal.c` expose them via `pg_gpu` vtable.
  - ops dispatch (CPU fallback preserved):
    `src/ops/elementwise.c:17` `try_bin_gpu`/`try_map_gpu` for ADD/SUB/MUL/DIV and EXP/LOG/SIN/COS/ERF/NEG/ABS/SQRT;
    `src/ops/activations.c:22` `try_map_gpu_act` for RELU/SIGMOID/TANH + `pg_softmax:95` GPU path;
    `src/ops/reduce.c:59` `try_sum_gpu` for SUM/MEAN via `pg_op_sum_axis`;
    `src/ops/matmul.c:50` `try_matmul_gpu`/`try_bmm_gpu` for matmul/bmm via `pg_gemm` on device buffers;
    `src/autograd/autograd.c:180` `try_accum_gpu` for broadcast grad accumulation via `pg_op_accum_scatter`.
    All paths allocate device buffers, `pg_copy_h2d`, `pg_op_*`, `pg_dev_sync`, `pg_copy_d2h`,
    free — on `PG_ERR_UNSUPPORTED` or `NULL` malloc they return `NULL` and the caller falls back to CPU loops,
    so `make BACKEND=cuda` on a machine without a driver still passes all tests.
  - autograd `backward` reuses the same ops, therefore runs on GPU automatically;
    `examples/train_xor.c` dispatches tanh/matmul/add/mul/mean through GPU kernels.
    Verified: `make BACKEND=cuda && ./build/train_xor` and `make BACKEND=metal && ./build/train_xor` both 4/4
    on CPU-only host via graceful fallback, and on T4 the per-op H2D/D2H path is green
    (bit-exact dyadic data). Zero-copy (device-resident tensors, no per-op copies) is deferred to M5
    as an optimization; current M3 uses per-op copies which keeps the hot loop logically on GPU.

- [ ] **M4 — HIP/ROCm** (when AMD hardware is available)
  dlopen libamdhip64.so, HSA path; M3 kernels ported via hipify.
  Interface is ready; estimate 1–2 days.

- [ ] **M5 — tail**
  Streams and pinned memory (async copies), index/gather/scatter,
  strided reductions. scan/sort/topk may stay on CPU indefinitely.

## Testing

- Existing suites run with `PG_DEVICE=gpu`: numeric gradient checks and
  the XOR example validate the new backend through the same public API.
- Loosen `allclose` tolerances for GPU (different summation order):
  rtol/atol ~1e-3 instead of 1e-4.
- One-command smoke script per backend (important for ephemeral Colab
  sessions).

## Risks

- No local GPU edit→run loop — iterations go through Colab/M4 only;
  batch edits into sessions.
- Own sgemm is slower than vendor libraries (see expectations above);
  risk of sinking time into tuning — correctness first, speed after M3.
- Metal: limited low-level control; shader debugging is peculiar.
- ROCm: check supported-GPU lists before starting M4.
