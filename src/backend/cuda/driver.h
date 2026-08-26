#ifndef PICOGRAD_BACKEND_CUDA_DRIVER_H
#define PICOGRAD_BACKEND_CUDA_DRIVER_H

#include "../backend.h"

#include <stddef.h>

typedef struct pg_cuda_drv {
    void *handle;
    int (*init)(unsigned flags);
    int (*device_get)(int *device, int ordinal);
    int (*ctx_create)(void **ctx, unsigned flags, int device);
    int (*mem_alloc)(unsigned long long *dptr, size_t nbytes);
    int (*mem_free)(unsigned long long dptr);
    int (*memcpy_h2d)(unsigned long long dst, const void *src, size_t nbytes);
    int (*memcpy_d2h)(void *dst, unsigned long long src, size_t nbytes);
    int (*memcpy_d2d)(unsigned long long dst, unsigned long long src,
                      size_t nbytes);
    int (*memset_u32)(unsigned long long dst, unsigned value, size_t count);
    int (*ctx_sync)(void);
    int (*module_load_data)(void **module, const void *image);
    int (*module_get_function)(void **func, void *module, const char *name);
    int (*launch_kernel)(void *func,
                         unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned shared_bytes, void *stream,
                         void **params, void **extra);
} pg_cuda_drv;

const pg_cuda_drv *pg_cuda_drv_get(pg_status *err);

#endif
