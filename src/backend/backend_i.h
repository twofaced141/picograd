#ifndef PICOGRAD_BACKEND_I_H
#define PICOGRAD_BACKEND_I_H

#include "backend.h"

typedef struct pg_backend_ops {
    const char *name;
    pg_status (*init)(void);
    void *(*malloc)(size_t nbytes);
    void (*free)(void *p);
    pg_status (*copy_h2d)(void *dst, const void *src, size_t nbytes);
    pg_status (*copy_d2h)(void *dst, const void *src, size_t nbytes);
    pg_status (*sync)(void);
    void (*gemm)(size_t m, size_t n, size_t k,
                 const float *a, size_t lda,
                 const float *b, size_t ldb,
                 float *c, size_t ldc);
} pg_backend_ops;

extern const pg_backend_ops pg_backend_cpu;

#if defined(PICOGRAD_BACKEND_CUDA)
extern const pg_backend_ops pg_backend_cuda;
#endif

#if defined(PICOGRAD_BACKEND_METAL)
extern const pg_backend_ops pg_backend_metal;
#endif

/* device kernel entry points, registered by the active gpu backend */
typedef struct pg_gpu_kernels {
    pg_status (*map)(float *, const float *, size_t, int);
    pg_status (*bin)(float *, const float *, const float *, size_t, int,
                     const pg_k_bin_args *);
    pg_status (*accum_gather)(float *, const float *, float,
                              const pg_k_strides *);
    pg_status (*accum_scatter)(float *, const float *, float,
                               const pg_k_strides *);
    pg_status (*sum_axis)(float *, const float *, float, size_t, size_t,
                          size_t, size_t);
    pg_status (*softmax)(float *, const float *, size_t, size_t, size_t);
    pg_status (*copy_strided)(float *, const float *, const pg_k_strides *);
    pg_status (*fill)(void *, size_t, float);
    pg_status (*copy_d2d)(void *, const void *, size_t);
} pg_gpu_kernels;

extern pg_gpu_kernels pg_gpu;

#endif
