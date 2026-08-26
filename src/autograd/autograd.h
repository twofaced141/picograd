#ifndef PICOGRAD_AUTOGRAD_H
#define PICOGRAD_AUTOGRAD_H

#include "../core/tensor.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pg_node {
    pg_tensor *value;
    pg_tensor *grad;
    size_t nparents;
    struct pg_node **parents;
    void *ctx;
    void (*backward)(struct pg_node *self);
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

pg_node *pg_ag_matmul(pg_node *a, pg_node *b);

pg_node *pg_ag_sum(pg_node *a, size_t axis, bool keepdim);
pg_node *pg_ag_mean(pg_node *a, size_t axis, bool keepdim);
pg_node *pg_ag_sum_all(pg_node *a);
pg_node *pg_ag_mean_all(pg_node *a);

pg_node *pg_ag_softmax(pg_node *a, size_t axis);

void pg_backward(pg_node *loss);

#ifdef __cplusplus
}
#endif

#endif
