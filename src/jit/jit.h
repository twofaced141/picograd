#ifndef PICOGRAD_JIT_H
#define PICOGRAD_JIT_H

#include "../core/tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ops that can be fused. Binary ops support broadcasting, unary are elementwise.
   STEP is x>0?1:0 for relu grad. */
typedef enum {
    PG_JIT_ADD = 0,
    PG_JIT_SUB,
    PG_JIT_MUL,
    PG_JIT_DIV,
    PG_JIT_NEG,
    PG_JIT_EXP,
    PG_JIT_LOG,
    PG_JIT_SQRT,
    PG_JIT_SIN,
    PG_JIT_COS,
    PG_JIT_ABS,
    PG_JIT_RELU,
    PG_JIT_SIGMOID,
    PG_JIT_TANH,
    PG_JIT_GELU,
    PG_JIT_ERF,
    PG_JIT_STEP, /* x > 0 ? 1 : 0  (relu grad) */
} pg_jit_op_t;

typedef struct pg_jit_graph pg_jit_graph;
typedef struct pg_jit_exe pg_jit_exe;

/* ---------- graph building ---------- */
pg_jit_graph *pg_jit_graph_new(void);
void pg_jit_graph_free(pg_jit_graph *g);

/* returns node id (>=0) or -1 on error */
int pg_jit_add_input(pg_jit_graph *g, size_t ndim, const size_t *shape);
int pg_jit_add_const(pg_jit_graph *g, float value); /* scalar broadcast */

int pg_jit_add(pg_jit_graph *g, int a, int b);
int pg_jit_sub(pg_jit_graph *g, int a, int b);
int pg_jit_mul(pg_jit_graph *g, int a, int b);
int pg_jit_div(pg_jit_graph *g, int a, int b);

int pg_jit_neg(pg_jit_graph *g, int a);
int pg_jit_exp(pg_jit_graph *g, int a);
int pg_jit_log(pg_jit_graph *g, int a);
int pg_jit_sqrt(pg_jit_graph *g, int a);
int pg_jit_sin(pg_jit_graph *g, int a);
int pg_jit_cos(pg_jit_graph *g, int a);
int pg_jit_abs(pg_jit_graph *g, int a);
int pg_jit_relu(pg_jit_graph *g, int a);
int pg_jit_sigmoid(pg_jit_graph *g, int a);
int pg_jit_tanh(pg_jit_graph *g, int a);
int pg_jit_gelu(pg_jit_graph *g, int a);
int pg_jit_erf(pg_jit_graph *g, int a);
int pg_jit_step(pg_jit_graph *g, int a); /* x > 0 ? 1 : 0 */

int pg_jit_add_op(pg_jit_graph *g, pg_jit_op_t op, const int *inputs, size_t ninputs);

void pg_jit_mark_output(pg_jit_graph *g, int id);
void pg_jit_clear_outputs(pg_jit_graph *g);

size_t pg_jit_num_inputs(const pg_jit_graph *g);
size_t pg_jit_num_outputs(const pg_jit_graph *g);
size_t pg_jit_num_nodes(const pg_jit_graph *g);

/* ---------- compilation & execution ---------- */
pg_jit_exe *pg_jit_compile(pg_jit_graph *g);
void pg_jit_exe_free(pg_jit_exe *exe);

/* inputs/outputs are arrays of pg_tensor* . Shapes must match graph.
 * Returns true on success.
 */
bool pg_jit_run(pg_jit_exe *exe, const pg_tensor **inputs, size_t ninputs,
                pg_tensor **outputs, size_t noutputs);

/* convenience: single output, allocates output tensor */
pg_tensor *pg_jit_run_single(pg_jit_exe *exe, const pg_tensor **inputs, size_t ninputs);

/* diagnostics */
const char *pg_jit_last_error(void);
size_t pg_jit_cache_size(void);
void pg_jit_cache_clear(void);
uint64_t pg_jit_graph_hash(const pg_jit_graph *g);

/* ---------- cache control ---------- */
void pg_jit_set_cache_enabled(bool enabled);
bool pg_jit_is_cache_enabled(void);

#ifdef __cplusplus
}
#endif

#endif
