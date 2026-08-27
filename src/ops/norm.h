#ifndef PICOGRAD_OPS_NORM_H
#define PICOGRAD_OPS_NORM_H

#include "../core/tensor.h"
#include <stdbool.h>

// LayerNorm over last dimension: y = (x - mean)/sqrt(var+eps) * weight + bias
// weight/bias may be NULL (no affine). If not NULL, shape must be [last_dim]
pg_tensor *pg_layernorm(const pg_tensor *x,
                        const pg_tensor *weight,
                        const pg_tensor *bias,
                        float eps);

// RMSNorm: y = x / sqrt(mean(x^2)+eps) * weight
pg_tensor *pg_rmsnorm(const pg_tensor *x,
                      const pg_tensor *weight,
                      float eps);

#endif
