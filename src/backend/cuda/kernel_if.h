#ifndef PICOGRAD_BACKEND_CUDA_KERNEL_IF_H
#define PICOGRAD_BACKEND_CUDA_KERNEL_IF_H

#include <stddef.h>

#define PG_MAX_NDIM_K 8

/* Layout mirrors (host <-> device code): all-unsigned structs, natural
 * alignment — safe to pass by value through cuLaunchKernel params.
 * Opcode enums live in backend.h. */

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned shape[PG_MAX_NDIM_K];
} pg_k_shape;

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned shape[PG_MAX_NDIM_K];
    unsigned sa[PG_MAX_NDIM_K];
    unsigned sb[PG_MAX_NDIM_K];
} pg_k_bin_args;

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned s[PG_MAX_NDIM_K];
} pg_k_strides;

#endif
