#ifndef PICOGRAD_OPT_CLIP_H
#define PICOGRAD_OPT_CLIP_H

#include "../autograd/autograd.h"
#include "../nn/module.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Clip gradients by global L2 norm.
// params: array of n parameter nodes (may contain NULL or nodes without grad -> skipped)
// max_norm: threshold (>0), eps: small value to avoid div-by-zero (e.g. 1e-6)
// Returns total norm before clipping (>=0). If norm > max_norm, scales each grad by max_norm/norm.
// Returns -1 on error (bad args). 0 if no grads.
// Example: pg_clip_grad_norm(params, n, 1.0f, 1e-6f);
float pg_clip_grad_norm(pg_node **params, size_t n, float max_norm, float eps);

// Convenience for pg_module
float pg_clip_grad_norm_module(pg_module *m, float max_norm, float eps);

// Clip each grad value into [-clip_value, +clip_value]
void pg_clip_grad_value(pg_node **params, size_t n, float clip_value);
void pg_clip_grad_value_module(pg_module *m, float clip_value);

// Compute total grad L2 norm without clipping (useful for logging)
float pg_grad_norm(pg_node **params, size_t n);
float pg_grad_norm_module(pg_module *m);

#ifdef __cplusplus
}
#endif

#endif
