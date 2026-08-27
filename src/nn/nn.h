#ifndef PICOGRAD_NN_NN_H
#define PICOGRAD_NN_NN_H

#include "../core/tensor.h"

typedef struct {
    size_t in_features;
    size_t out_features;
    pg_tensor *weight;
    pg_tensor *bias;
} pg_linear;

pg_linear *pg_linear_new(size_t in_features, size_t out_features);
void pg_linear_free(pg_linear *layer);
pg_tensor *pg_linear_forward(const pg_linear *layer, const pg_tensor *x);
// fused variant: matmul + bias + relu in one kernel
pg_tensor *pg_linear_forward_relu(const pg_linear *layer, const pg_tensor *x);

#endif
