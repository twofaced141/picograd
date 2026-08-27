# picograd

A tiny tensor library in C11 with reverse-mode autograd and SGD, running on the CPU.

## Features

- Tensors up to 8 dimensions, row-major, `float32`
- Ops: elementwise (broadcasting), matmul/bmm/tensordot, reductions, activations,
  indexing/gather/scatter, cumsum/sort/topk
- Reverse-mode autograd over a dynamically built computation graph
- Optimizer: SGD with momentum, dampening, weight decay and Nesterov variant
- Hand-written AVX2 / AVX-512 GEMM microkernels
- **JIT compilation** — tracing + fusion of elementwise chains into a single
  `cc -O3 -fPIC -shared` kernel (`dlopen` at runtime), with broadcast support
  and cross-run cache (`src/jit`)

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
eager: ~900 ms (4 passes)
jit  : ~140 ms (1 pass)  ~6.5x
```

See `examples/jit_demo.c` and `examples/train_xor_jit.c`, tests `tests/test_jit.c`.

MVP limitations: elementwise fusion only, single loop shape (all outputs share `same numel`), `matmul/reduce` remain separate kernels; JIT is an inference/forward optimization, backward is still eager.

## Layout

```
src/core        tensors, shapes, RNG
src/ops         elementwise, matmul, reduce, activations, index, scan
src/autograd    computation graph, backward pass
src/opt         SGD
src/jit         JIT tracing, fusion, codegen + cache (cc/dlopen)
src/backend     AVX2/AVX-512 GEMM kernels + CUDA/HIP/Metal dispatch
src/nn          minimal nn layers (plain tensors)
tests           unit tests incl. numeric gradient checks
examples        end-to-end training demos
```

## License

MIT
