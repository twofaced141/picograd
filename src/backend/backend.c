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

pg_gpu_kernels pg_gpu;

static const pg_backend_ops *ops_for(pg_devtype dev)
{
    switch (dev) {
    case PG_DEV_CPU:
        return &pg_backend_cpu;
#if defined(PICOGRAD_BACKEND_CUDA)
    case PG_DEV_CUDA:
        return &pg_backend_cuda;
#endif
#if defined(PICOGRAD_BACKEND_METAL)
    case PG_DEV_METAL:
        return &pg_backend_metal;
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

pg_status pg_op_map(float *out, const float *src, size_t n, int op)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.map)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.map(out, src, n, op);
}

pg_status pg_op_bin(float *out, const float *a, const float *b, size_t n,
                    int op, const pg_k_bin_args *args)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.bin)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.bin(out, a, b, n, op, args);
}

pg_status pg_op_accum_gather(float *dst, const float *src, float scale,
                             const pg_k_strides *args)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.accum_gather)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.accum_gather(dst, src, scale, args);
}

pg_status pg_op_accum_scatter(float *dst, const float *src, float scale,
                              const pg_k_strides *args)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.accum_scatter)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.accum_scatter(dst, src, scale, args);
}

pg_status pg_op_sum_axis(float *out, const float *src, float scale,
                         size_t outer, size_t len, size_t inner,
                         size_t ostride)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.sum_axis)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.sum_axis(out, src, scale, outer, len, inner, ostride);
}

pg_status pg_op_softmax(float *out, const float *src, size_t outer,
                        size_t len, size_t inner)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.softmax)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.softmax(out, src, outer, len, inner);
}

pg_status pg_op_copy_strided(float *dst, const float *src,
                             const pg_k_strides *args)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.copy_strided)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.copy_strided(dst, src, args);
}

pg_status pg_op_fill(void *p, size_t nbytes, float v)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.fill)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.fill(p, nbytes, v);
}

pg_status pg_op_copy_d2d(void *dst, const void *src, size_t nbytes)
{
    if (g_device == PG_DEV_CPU || !pg_gpu.copy_d2d)
        return PG_ERR_UNSUPPORTED;
    return pg_gpu.copy_d2d(dst, src, nbytes);
}

/* GPU buffer helper */
pg_dev_buf pg_dev_buf_new(size_t nbytes)
{
    pg_dev_buf buf = {0};
    if (!nbytes)
        return buf;
    buf.ptr = pg_dev_malloc(nbytes);
    buf.nbytes = nbytes;
    return buf;
}

void pg_dev_buf_free(pg_dev_buf *buf)
{
    if (buf->ptr) {
        pg_dev_free(buf->ptr);
        buf->ptr = NULL;
    }
    buf->nbytes = 0;
}

/* GPU exec helper */
pg_dev_exec pg_dev_exec_begin(pg_tensor *result)
{
    pg_dev_exec exec = {true, result};
    return exec;
}

bool pg_dev_exec_check(pg_dev_exec *exec, bool condition)
{
    if (!condition)
        exec->ok = false;
    return exec->ok;
}

void pg_dev_exec_end(pg_dev_exec *exec)
{
    (void)exec;
}
