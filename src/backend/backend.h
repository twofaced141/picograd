#ifndef PICOGRAD_BACKEND_H
#define PICOGRAD_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PG_DEV_CPU = 0,
    PG_DEV_CUDA,
    PG_DEV_METAL,
    PG_DEV_HIP,
} pg_devtype;

typedef enum {
    PG_OK = 0,
    PG_ERR_UNSUPPORTED = -1,
    PG_ERR_ALLOC = -2,
    PG_ERR_COPY = -3,
    PG_ERR_SYNC = -4,
    PG_ERR_GEMM = -5,
} pg_status;

pg_status pg_set_device(pg_devtype dev);
pg_devtype pg_get_device(void);

void *pg_dev_malloc(size_t nbytes);
void pg_dev_free(void *p);

pg_status pg_copy_h2d(void *dst, const void *src, size_t nbytes);
pg_status pg_copy_d2h(void *dst, const void *src, size_t nbytes);
pg_status pg_dev_sync(void);

void pg_gemm(size_t m, size_t n, size_t k,
             const float *a, size_t lda,
             const float *b, size_t ldb,
             float *c, size_t ldc);

#ifdef __cplusplus
}
#endif

#endif
