# JIT in picograd

## Idea

A tracing JIT for the CPU that fuses chains of elementwise operations into a single compiled kernel.

Motivation: eager execution runs each operation as a separate memory pass:

```
t0 = add(a,b)   // pass 1: read a,b -> write t0
t1 = mul(t0,c)  // pass 2: read t0,c -> write t1
t2 = relu(t1)   // pass 3: read t1 -> write out
```

3 allocations, 3 loops, 6 reads/writes. With fusion:

```
for i in 0..N-1:
  v0 = a[i]; v1 = b[i]; v2=c[i];
  t0 = v0+v1; t1=t0*v2; out[i]= t1>0? t1:0;
```

1 pass, 0 intermediate buffers, better locality and the compiler can auto-vectorize.

## Architecture

```
pg_jit_graph --(emit C)--> /tmp/picograd_jit_*.c --(cc -O3 -fPIC -shared)--> /tmp/*.so --(dlopen)--> kernel
      |                              |
      +-- hash (FNV-1a) --> cache (linked list)
```

### 1. Tracing

`pg_jit_graph` — a DAG.

- `pg_jit_add_input(ndim, shape)` — input
- `pg_jit_add_const(value)` — scalar (literal, not memory)
- `pg_jit_add/mul/...` — create a `JNODE_OP` node with computed `out_shape` via `bcast_shape2` (same rules as `pg_bcast_shape`) and `numel/strides`.

All nodes store `ndim/shape/stride/numel`. Nodes are appended in creation order — topological order if input ids < output ids (checked).

`pg_jit_mark_output(id)` — marks an output.

### 2. Code generation

`emit_c_code()` generates:

```c
#include <math.h>
#include <stddef.h>
void pg_jit_kernel(float** outs, const float** ins){
  const float* in0 = ins[0]; ...
  float* out0 = outs[0]; ...
  for(size_t i=0;i<N;i++){
    size_t tmp=i;
    size_t c1 = tmp % shape[1]; tmp/=shape[1];
    size_t c0 = tmp;
    size_t off_in0 = c0*stride0 + c1*stride1; // for broadcast
    float v0 = in0[off_in0];
    float v1 = in1[off_in1];
    float v2 = (v0 + v1);
    float v3 = expf(v2);
    out0[i]=v3;
  }
}
```

- For direct matches `shape_in == shape_out` => `off=i` without divisions.
- For broadcast — `c[d]` via `%`/`/` and `off=Σ c[d]*bstride[d]`. `bstride` is computed from `compute_bcast_strides`.
- Constants — `float vX = 2.5F;`
- Multi-output with identical `numel` — one loop, multiple `outY[i]=vZ;`. Different `numel` — compile error (MVP).

Op → C mapping:

| op | C |
|---|---|
| add/sub/mul/div | `a+b` etc |
| neg | `-a` |
| exp/log/sqrt/sin/cos/abs | `expf(a)` etc |
| relu | `a>0?a:0` |
| sigmoid | `1/(1+expf(-a))` |
| tanh | `tanhf(a)` |
| gelu | `0.5*a*(1+erff(a*0.7071))` |
| erf | `erff(a)` |

### 3. Compilation

- Temp paths `/tmp/picograd_jit_<pid>_<cnt>_<rand>.c/.so` (unique, not `mkstemp` so the `.so` stays for the cache).
- Compiler discovery: `which cc`/`gcc`/`clang` → `cc -O3 -ffast-math -fPIC -shared -o so c -lm 2>&1`.
- `dlopen(so, RTLD_NOW)` + `dlsym("pg_jit_kernel")`.
- Errors → `g_last_err`, `pg_jit_last_error()`.

Requirements: `cc` in `PATH`, `-ldl` (already in Makefile).

### 4. Cache

- Key — `FNV-1a` over `nnodes/ninputs/noutputs + each node (kind,op,inputs,ndim,shape,const)` + `inputs/outputs` arrays.
- On `pg_jit_compile(g)` first search the linked list `g_cache_head`. On hit — `dlopen` the same `.so` (new handle, refcnt increment), no recompilation.
- On miss — compile + insert into cache (`graph_clone` for `graph_equal` on collisions).
- `pg_jit_cache_size()` / `pg_jit_cache_clear()` (unlink all `.so/.c`).
- `pg_jit_exe_free()` — `dlclose` + unlink only if the path is not in the cache (cache owns files until `clear`).

Thread safety for MVP — none (single thread).

### 5. Execution

`pg_jit_run(exe, ins, nins, outs, nouts)`:

- validates `ndim/shape` of inputs/outputs vs `exe->in_shape/out_shape`
- `float* out_ptrs[16]; const float* in_ptrs[16];` → `exe->kernel(out_ptrs, in_ptrs)`

`pg_jit_run_single()` — allocates a `pg_tensor` for the single output.

## Usage

See `examples/jit_demo.c`, `examples/train_xor_jit.c`, `tests/test_jit.c`.

## Future extensions

- Add `matmul` to JIT: generate a call to `pg_cpu_gemm` or a naive loop (but loses AVX).
- Backward JIT: trace the `pg_node` graph, generate a fused backward kernel.
- `pg_jit_trace` mode: intercept `pg_add/mul/...` via a thread-local flag so `forward()` builds a `pg_jit_graph` without changes.
- `PG_JIT_DEBUG=1` — dump generated `.c`.
- Support different `numel` multi-output via separate loops.

## Files

- `src/jit/jit.h` — public API
- `src/jit/jit.c` — implementation (~700 lines)
- `tests/test_jit.c` — 12 tests, ~300 lines
- `examples/jit_demo.c` — demo + 6.5x bench
- `examples/train_xor_jit.c` — XOR with JIT bias+act
