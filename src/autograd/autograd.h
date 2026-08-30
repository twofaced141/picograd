#ifndef PICOGRAD_AUTOGRAD_H
#define PICOGRAD_AUTOGRAD_H

#include "../core/tensor.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PG_AG_OP_NONE = 0,
    PG_AG_OP_ADD,
    PG_AG_OP_SUB,
    PG_AG_OP_MUL,
    PG_AG_OP_DIV,
    PG_AG_OP_NEG,
    PG_AG_OP_EXP,
    PG_AG_OP_LOG,
    PG_AG_OP_SQRT,
    PG_AG_OP_SIN,
    PG_AG_OP_COS,
    PG_AG_OP_RELU,
    PG_AG_OP_SIGMOID,
    PG_AG_OP_TANH,
    PG_AG_OP_MATMUL,
    PG_AG_OP_SUM,
    PG_AG_OP_MEAN,
    PG_AG_OP_SOFTMAX,
    PG_AG_OP_TRANSPOSE,
    PG_AG_OP_LAYERNORM,
    PG_AG_OP_RMSNORM,
    PG_AG_OP_CONV2D,
    PG_AG_OP_EMBEDDING,
    PG_AG_OP_DROPOUT,
    PG_AG_OP_CROSS_ENTROPY,
    PG_AG_OP_MSE,
    PG_AG_OP_BCE,
    PG_AG_OP_POW,
    PG_AG_OP_ABS,
    PG_AG_OP_CLAMP,
    PG_AG_OP_GELU,
    PG_AG_OP_LEAKY_RELU,
    PG_AG_OP_ERF,
    PG_AG_OP_BATCHNORM,
    PG_AG_OP_COUNT
} pg_ag_op_t;

typedef struct pg_node {
    pg_tensor *value;
    pg_tensor *grad;
    size_t nparents;
    struct pg_node **parents;
    void *ctx;
    void (*backward)(struct pg_node *self);
    pg_ag_op_t ag_op;
    bool requires_grad;
    unsigned refs;
    unsigned char mark;
} pg_node;

pg_node *pg_var_from_data(size_t ndim, const size_t *shape, const float *data,
                          bool requires_grad);
pg_node *pg_var_from_tensor(const pg_tensor *t, bool requires_grad);
pg_node *pg_var_zeros(size_t ndim, const size_t *shape, bool requires_grad);
pg_node *pg_var_ones(size_t ndim, const size_t *shape, bool requires_grad);
pg_node *pg_var_full(size_t ndim, const size_t *shape, float value,
                     bool requires_grad);
pg_node *pg_var_uniform(size_t ndim, const size_t *shape, float low, float high,
                        bool requires_grad);
pg_node *pg_var_normal(size_t ndim, const size_t *shape, float mean,
                       float stddev, bool requires_grad);
pg_node *pg_var_scalar(float value, bool requires_grad);

pg_node *pg_node_retain(pg_node *n);
void pg_node_free(pg_node *n);

static inline pg_tensor *pg_node_value(const pg_node *n)
{
    return n ? n->value : NULL;
}

static inline pg_tensor *pg_node_grad(const pg_node *n)
{
    return n ? n->grad : NULL;
}

pg_node *pg_ag_add(pg_node *a, pg_node *b);
pg_node *pg_ag_sub(pg_node *a, pg_node *b);
pg_node *pg_ag_mul(pg_node *a, pg_node *b);
pg_node *pg_ag_div(pg_node *a, pg_node *b);

pg_node *pg_ag_neg(pg_node *a);
pg_node *pg_ag_exp(pg_node *a);
pg_node *pg_ag_log(pg_node *a);
pg_node *pg_ag_sqrt(pg_node *a);
pg_node *pg_ag_sin(pg_node *a);
pg_node *pg_ag_cos(pg_node *a);

pg_node *pg_ag_relu(pg_node *a);
pg_node *pg_ag_sigmoid(pg_node *a);
pg_node *pg_ag_tanh(pg_node *a);

pg_node *pg_ag_pow(pg_node *a, pg_node *b);
pg_node *pg_ag_abs(pg_node *a);
pg_node *pg_ag_clamp(pg_node *a, float lo, float hi);
pg_node *pg_ag_gelu(pg_node *a);
pg_node *pg_ag_leaky_relu(pg_node *a, float alpha);
pg_node *pg_ag_erf(pg_node *a);

pg_node *pg_ag_matmul(pg_node *a, pg_node *b);

pg_node *pg_ag_sum(pg_node *a, size_t axis, bool keepdim);
pg_node *pg_ag_mean(pg_node *a, size_t axis, bool keepdim);
pg_node *pg_ag_sum_all(pg_node *a);
pg_node *pg_ag_mean_all(pg_node *a);

pg_node *pg_ag_softmax(pg_node *a, size_t axis);

pg_node *pg_ag_transpose(pg_node *a, size_t axis0, size_t axis1);

pg_node *pg_ag_layernorm(pg_node *x, pg_node *weight, pg_node *bias, float eps);
pg_node *pg_ag_rmsnorm(pg_node *x, pg_node *weight, float eps);
pg_node *pg_ag_batchnorm(pg_node *x, pg_node *weight, pg_node *bias, float eps);
pg_node *pg_ag_batchnorm2d(pg_node *x, pg_node *weight, pg_node *bias,
                           pg_tensor *running_mean, pg_tensor *running_var,
                           float eps, float momentum, bool training);

/* ---- extra nn layers (training) ---- */
/* NCHW 2D convolution. x:[N,Cin,H,W], w:[Cout,Cin,kh,kw], b:[Cout] (may be NULL). */
pg_node *pg_ag_conv2d(pg_node *x, pg_node *w, pg_node *b,
                      size_t kh, size_t kw, int stride, int padding);

/* Embedding lookup. weight:[V,E] (learned), indices: integer class ids (not differentiated). */
pg_node *pg_ag_embedding(pg_node *weight, const pg_tensor *indices);

/* Dropout. training=false is an identity (eval mode). */
pg_node *pg_ag_dropout(pg_node *x, float p, bool training);

/* ---- loss functions (training) ---- */
/* Cross-entropy with integer targets. `logits` has shape [..., C],
 * `targets` has length R = numel/C (one class index per leading row).
 * Numerically stable (log-sum-exp). grad = softmax - onehot (mean-scaled). */
pg_node *pg_ag_cross_entropy(pg_node *logits, const size_t *targets, size_t n, bool mean);

/* Mean squared error. grad = 2*(pred-target) (mean-scaled). */
pg_node *pg_ag_mse(pg_node *pred, pg_node *target, bool mean);

/* Binary cross-entropy with logits (targets in [0,1]).
 * Stable form: max(z,0) - z*t + log(1+exp(-|z|)). grad = sigmoid(z) - t. */
pg_node *pg_ag_bce_with_logits(pg_node *logits, const float *targets, size_t n, bool mean);

void pg_backward(pg_node *loss);

// detach / no_grad
pg_node *pg_node_detach(pg_node *n);
bool pg_autograd_is_grad_enabled(void);
void pg_autograd_set_grad_enabled(bool enabled);
void pg_no_grad_push(void);
void pg_no_grad_pop(void);
// helper macros
#define PG_NO_GRAD_BEGIN do{ pg_no_grad_push(); }while(0)
#define PG_NO_GRAD_END do{ pg_no_grad_pop(); }while(0)

/* ---- JIT autograd ---- */
void pg_autograd_set_jit(bool enabled);
bool pg_autograd_is_jit_enabled(void);
bool pg_backward_jit(pg_node *loss); /* try JIT path, false = fallback */
const char *pg_autograd_last_error(void);
/* diagnostics */
bool pg_autograd_last_was_jit(void);
size_t pg_autograd_jit_hits(void);
size_t pg_autograd_jit_fallbacks(void);

#ifdef __cplusplus
}
#endif

#endif
