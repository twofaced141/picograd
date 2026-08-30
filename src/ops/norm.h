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

// BatchNorm: per-channel normalization over N*spatial
// x: [N, C, ...] (ndim >=2, C = shape[1]), weight/bias: [C] or NULL, eps>0
// Simple training-only version (computes batch stats each forward)
pg_tensor *pg_batchnorm(const pg_tensor *x,
                        const pg_tensor *weight,
                        const pg_tensor *bias,
                        float eps);

// Full BatchNorm2d with running stats (inference support)
// running_mean/var: [C] mutable buffers updated in training when not NULL
// momentum in [0,1] (typical 0.1), training=true uses batch stats else running
pg_tensor *pg_batchnorm2d(const pg_tensor *x,
                          const pg_tensor *weight,
                          const pg_tensor *bias,
                          pg_tensor *running_mean,
                          pg_tensor *running_var,
                          float eps, float momentum, bool training);

#endif
