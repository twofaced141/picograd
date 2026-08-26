# picograd

A tiny tensor library in C11 with reverse-mode autograd and SGD, running on the CPU.

## Features

- Tensors up to 8 dimensions, row-major, `float32`
- Ops: elementwise (broadcasting), matmul/bmm/tensordot, reductions, activations,
  indexing/gather/scatter, cumsum/sort/topk
- Reverse-mode autograd over a dynamically built computation graph
- Optimizer: SGD with momentum, dampening, weight decay and Nesterov variant
- Hand-written AVX2 / AVX-512 GEMM microkernels

## Build

```
make            # builds libpicograd.a, tests and examples into build/
make test       # run the test suite
make examples   # build and run examples (e.g. XOR training)
make bench      # GEMM benchmark against MKL (requires Intel oneAPI)
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

## Layout

```
src/core        tensors, shapes, RNG
src/ops         elementwise, matmul, reduce, activations, index, scan
src/autograd    computation graph, backward pass
src/opt         SGD
src/backend     AVX2/AVX-512 GEMM kernels
src/nn          minimal nn layers (plain tensors)
tests           unit tests incl. numeric gradient checks
examples        end-to-end training demos
```

## License

MIT
