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

- [ ] **M1 — CUDA minimum, own kernels** (tested on Google Colab T4)
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
  Left to validate on real hardware (Colab):
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

- [ ] **M2 — Metal minimum** (tested on M4)
  Obj-C++ / metal-cpp shim: device, command queue, buffers.
  GEMM — own MSL kernel (same tiled scheme; MPS rejected for the same
  reason as cuBLAS). Same `pg_gemm` entry point, same test contract.
  Metal bonus: shaders are compiled by the system from source at runtime —
  no separate build pipeline needed.

- [ ] **M3 — elementwise/reduce/activation kernels**
  Map/reduce kernels (macro-generated from a single op declaration) for
  `.cu`→PTX and `.metal`; softmax/log_softmax. After that autograd works
  on-device automatically (backward reuses ops), and the XOR example runs
  fully on GPU with no copies in the hot loop.

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
