#ifndef PICOGRAD_OPS_ACTIVATIONS_H
#define PICOGRAD_OPS_ACTIVATIONS_H

#include "../core/tensor.h"

pg_tensor *pg_relu(const pg_tensor *a);
pg_tensor *pg_leaky_relu(const pg_tensor *a, float alpha);
pg_tensor *pg_sigmoid(const pg_tensor *a);
pg_tensor *pg_tanh(const pg_tensor *a);
pg_tensor *pg_gelu(const pg_tensor *a);

pg_tensor *pg_softmax(const pg_tensor *t, size_t axis);
pg_tensor *pg_log_softmax(const pg_tensor *t, size_t axis);

#endif
