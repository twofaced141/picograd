# picograd

[![CI](https://github.com/twofaced141/picograd/actions/workflows/ci.yml/badge.svg)](https://github.com/twofaced141/picograd/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tiny tensor library in C11 with reverse-mode autograd and SGD — CPU by default with optional GPU backends (CUDA, HIP/ROCm, Metal) via `dlopen` at runtime.

## Features

- Tensors up to 8 dimensions, row-major, `float32` + mixed-precision `float16`/`bfloat16` storage with `float32` accumulation (`src/core/dtype.h`, `convert.h`)
- Ops: elementwise (broadcasting), matmul/bmm/tensordot (dtype-aware, `f16`/`bf16` via `pg_gemm_ex`), reductions, activations,
  indexing/gather/scatter, cumsum/sort/topk
- Reverse-mode autograd over a dynamically built computation graph, **JIT-accelerated for elementwise chains** (fused backward, fallback to eager for matmul/reduce), matmul backward uses `hgemm`/`bgemm` (gradients `f32`)
- Optimizer: SGD with momentum, dampening, weight decay and Nesterov variant
- Hand-written AVX2 / AVX-512 GEMM microkernels + mixed-precision kernels: `AVX2` `vcvtph2ps`/`vfmadd` (1.3x BW), `AVX512FP16` `vfmadd231ph` 32x32, `AVX512BF16` `vcvtneps2bf16`+`vdpbf16ps`, `AMX-BF16` `tileloadd`/`tdpbf16ps` 16x32, `ARM NEON` `vld1q_f16`/`vfmaq` 16x8, `SVE2-BF16` `bfmmla`/`bfdot`, GPU `WMMA` `mma.m16n8k16`/`cp.async` (CUDA), `rocWMMA`/`__hip_half` (HIP), `simdgroup_matrix` (Metal)
- **JIT compilation** — tracing + fusion of elementwise chains into a single
  `cc -O3 -fPIC -shared` kernel (`dlopen` at runtime), with broadcast support
  and cross-run cache (`src/jit`); **covers both forward and backward (autograd)** —
  elementwise backward chains are fused into one kernel (e.g., `sub->mul->tanh` backward in one pass)

## Build

```
make            # builds libpicograd.a, tests and examples into build/
make test       # run the test suite
make examples   # build and run examples (e.g. XOR training)
make bench      # GEMM benchmark against MKL (requires Intel oneAPI)
```

GPU backends (zero toolkit at build time, dlopen at runtime):

```
make BACKEND=cuda            # NVIDIA via CUDA Driver API + embedded PTX
make BACKEND=hip             # AMD via HIP + hiprtc JIT (alias: rocm)
make BACKEND=metal           # Apple Metal (Darwin) / stub elsewhere
make BACKEND=cuda test       # tests fall back to CPU when no GPU
make BACKEND=hip bench-gpu   # GEMM benchmark on AMD GPU
PG_HIP_DEBUG=1 ./build/test_backend  # verbose HIP loader
```

## Usage

```c
#include "src/autograd/autograd.h"
#include "src/opt/sgd.h"

float xd[4] = {1, -2, 3, -4};
pg_node *x = pg_var_from_data(1, (size_t[]){4}, xd, true);
pg_node *w = pg_var_uniform(1, (size_t[]){4}, -1.0f, 1.0f, true);

pg_node *y = pg_ag_mul(x, w);
pg_node *loss = pg_ag_sum_all(y);

pg_backward(loss);                      /* grads land in node->grad */

pg_sgd_cfg cfg = pg_sgd_cfg_default();
cfg.lr = 0.1f;
pg_sgd *opt = pg_sgd_new(&cfg);
pg_sgd_add_param(opt, w);
pg_sgd_step(opt);                       /* w -= lr * grad */

pg_node_free(loss);
pg_node_free(y);
pg_node_free(w);
pg_node_free(x);
pg_sgd_free(opt);
```

### Mixed precision (f16/bf16 storage + f32 accum)

```c
#include "src/core/dtype.h"
#include "src/core/convert.h"

size_t sh[2]={4,8};
pg_tensor *a = pg_tensor_empty_dtype(PG_DTYPE_F16, 2, sh);
pg_tensor *b = pg_tensor_empty_dtype(PG_DTYPE_F16, 2, (size_t[]){8,4});
// fill via convert
for(size_t i=0;i<a->numel;i++) a->data_u16[i]=pg_f32_to_f16_scalar(1.0f);
pg_tensor *c = pg_matmul(a,b); // c is F32 (f32 accum), uses hgemm (AVX2 vcvtph2ps / AVX512FP16 / AMX)
pg_gemm_ex(PG_DTYPE_F16, 4,4,8, a->data_raw, 8, b->data_raw, 4, c->data, 4);
```

`ld*` are in elements. Dispatch: `AMX-BF16 > AVX512FP16/BF16 > AVX2 cvt > generic` (x86), `SVE2+BF16`/`NEON fp16` (ARM), `WMMA` (GPU, 312 TFLOPS).

Ownership rule: every function returning a `pg_node *` hands you a new reference;
operands are borrowed, not consumed. Intermediate nodes must be released with
`pg_node_free` once they are no longer needed. Nodes returned by an op keep their
parents alive, so freeing a whole graph is just freeing its output.

A full MLP training loop lives in `examples/train_xor.c`.

## JIT compilation

`src/jit` implements a lightweight **tracing JIT** for the CPU:

- **Tracing** — `pg_jit_graph` builds a DAG from elementwise ops (`add/sub/mul/div/neg/exp/log/sqrt/sin/cos/abs/relu/sigmoid/tanh/gelu/erf`) with broadcasting support (same rules as `src/ops/common.h`).
- **Fusion** — the entire graph is fused into a single `for (i < numel)` loop: one memory pass instead of N.
- **Codegen** — generates `C` code (`#include <math.h>`), compiles it with `cc -O3 -ffast-math -fPIC -shared -o /tmp/picograd_jit_*.so` and loads it via `dlopen`/`dlsym`. Requires `cc`/`gcc`/`clang` in `PATH` and `-ldl` (already in `LDLIBS`).
- **Cache** — `FNV-1a` hash of topology + shapes; recompiling the same graph is a cache hit with no recompilation (`pg_jit_cache_size()`, `pg_jit_cache_clear()`).
- **Broadcast** — inside the kernel `off = Σ c[d]*stride[d]` via `c[d] = tmp % shape[d]`; for direct matches `off = i`.

```c
#include "src/jit/jit.h"

pg_jit_graph *g = pg_jit_graph_new();
int a = pg_jit_add_input(g, 2, (size_t[]){4,8});
int b = pg_jit_add_input(g, 1, (size_t[]){8});
int c = pg_jit_add(g, a, b);          // broadcast [4,8] + [8]
int h = pg_jit_tanh(g, c);
pg_jit_mark_output(g, h);

pg_jit_exe *exe = pg_jit_compile(g);  // first time compiles, second is cache hit
pg_tensor *out = pg_jit_run_single(exe, (const pg_tensor*[]){ta, tb}, 2);
// or multi-output:
pg_tensor *outs[2] = {out_add, out_mul};
pg_jit_run(exe, ins, 2, outs, 2);

pg_jit_exe_free(exe);
pg_jit_graph_free(g);
pg_jit_cache_clear();
```

Benchmark for fused `[64,64]` `(a+b)*c -> relu -> tanh` (10000 iterations):

```
forward  eager: ~900 ms (4 passes)  jit: ~140 ms (1 pass)  ~6.5x
backward eager: ~540 ms (4 passes)  jit: ~1140 ms (1 pass, small) -> 0.5x (overhead dominates)
         large [512,512] eager: ~9700 ms  jit: ~5000 ms  ~1.9x (compute dominates)
```

See `examples/jit_demo.c` and `examples/train_xor_jit.c`, tests `tests/test_jit.c` and `tests/test_jit_autograd` (implicit via `test_autograd`).

MVP limitations: elementwise fusion only, single loop shape for pure elementwise graphs (all intermediate shapes equal, broadcast inputs allowed), `matmul/reduce` remain separate kernels; **JIT now covers both forward and backward** for elementwise chains (fused forward + fused backward), with suffix-fusion for mixed graphs (e.g., `mse` loss tail `sub->mul->mean` fused, `add(bias)` fused) and automatic fallback to eager for other ops. Pure elementwise graphs run fully JIT; mixed graphs (e.g., XOR MLP) run hybrid (JIT for elementwise suffix, eager for matmul). Enable/disable via `pg_autograd_set_jit(bool)` (`pg_autograd_is_jit_enabled()`, `pg_autograd_last_was_jit()`, `pg_autograd_jit_hits()`).

## Layout

```
src/core        tensors (dtype-aware void* + 64B pool), shapes, RNG, dtype/convert
src/ops         elementwise, matmul (mixed f16/bf16->f32, fused bias+act), reduce, activations, index, scan
src/autograd    computation graph, backward pass (hgemm/bgemm for grads)
src/opt         SGD
src/jit         JIT tracing, fusion, codegen + cache (cc/dlopen)
src/backend     GEMM: AVX2/AVX-512/AMX + NEON/SVE + CUDA WMMA/HIP rocWMMA/Metal simdgroup dispatch
src/nn          minimal nn layers (plain tensors)
tests           unit tests incl. numeric gradient checks (f16 tol 1e-2)
examples        end-to-end training demos
```

## License

MIT
