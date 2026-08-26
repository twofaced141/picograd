#include "backend_i.h"

#include "cpu/gemm.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void *cpu_malloc(size_t nbytes)
{
    return malloc(nbytes);
}

static pg_status cpu_copy(void *dst, const void *src, size_t nbytes)
{
    memcpy(dst, src, nbytes);
    return PG_OK;
}

const pg_backend_ops pg_backend_cpu = {
    .name = "cpu",
    .init = NULL,
    .malloc = cpu_malloc,
    .free = free,
    .copy_h2d = cpu_copy,
    .copy_d2h = cpu_copy,
    .sync = NULL,
    .gemm = NULL,
};

static pg_devtype g_device = PG_DEV_CPU;

static const pg_backend_ops *ops_for(pg_devtype dev)
{
    switch (dev) {
    case PG_DEV_CPU:
        return &pg_backend_cpu;
#if defined(PICOGRAD_BACKEND_CUDA)
    case PG_DEV_CUDA:
        return &pg_backend_cuda;
#endif
    default:
        return NULL;
    }
}

pg_status pg_set_device(pg_devtype dev)
{
    const pg_backend_ops *o = ops_for(dev);
    if (!o)
        return PG_ERR_UNSUPPORTED;

    if (o->init) {
        pg_status st = o->init();
        if (st != PG_OK)
            return st;
    }

    g_device = dev;
    return PG_OK;
}

pg_devtype pg_get_device(void)
{
    return g_device;
}

void *pg_dev_malloc(size_t nbytes)
{
    const pg_backend_ops *o = ops_for(g_device);
    return o ? o->malloc(nbytes) : NULL;
}

void pg_dev_free(void *p)
{
    const pg_backend_ops *o = ops_for(g_device);
    if (o && p)
        o->free(p);
}

pg_status pg_copy_h2d(void *dst, const void *src, size_t nbytes)
{
    const pg_backend_ops *o = ops_for(g_device);
    if (!o)
        return PG_ERR_UNSUPPORTED;
    return o->copy_h2d(dst, src, nbytes);
}

pg_status pg_copy_d2h(void *dst, const void *src, size_t nbytes)
{
    const pg_backend_ops *o = ops_for(g_device);
    if (!o)
        return PG_ERR_UNSUPPORTED;
    return o->copy_d2h(dst, src, nbytes);
}

pg_status pg_dev_sync(void)
{
    const pg_backend_ops *o = ops_for(g_device);
    if (!o)
        return PG_ERR_UNSUPPORTED;
    return o->sync ? o->sync() : PG_OK;
}

void pg_gemm(size_t m, size_t n, size_t k,
             const float *a, size_t lda,
             const float *b, size_t ldb,
             float *c, size_t ldc)
{
    const pg_backend_ops *o = ops_for(g_device);
    assert(o);

    if (o->gemm) {
        o->gemm(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }

    pg_cpu_gemm(m, n, k, a, lda, b, ldb, c, ldc);
}
