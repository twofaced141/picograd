#include "../backend_i.h"
#include "driver.h"

#include "sgemm_ptx.h"
#include "ops_ptx.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BTILE 64
#define TPT 16
#define PG_CUDA_THREADS 256

static void *g_ctx;
static void *g_module;
static void *g_fn_sgemm;
static void *g_ops_module;
static void *g_fn_map;
static void *g_fn_bin;
static void *g_fn_accum_scatter;
static void *g_fn_sum_axis;
static void *g_fn_softmax;
static void *g_fn_copy_strided;

static void cuda_register_gpu(void);

static pg_status cuda_init(void)
{
    static pg_status cached = PG_ERR_UNSUPPORTED;
    static int ok = 0;

    if (ok)
        return cached;

    int debug = getenv("PG_CUDA_DEBUG") != NULL;
    pg_status err = PG_OK;
    const pg_cuda_drv *drv = pg_cuda_drv_get(&err);
    if (!drv) {
        if (debug) fprintf(stderr, "picograd/cuda: driver get failed %d\n", err);
        cached = err;
        return cached;
    }

    int device;
    int rc;
    rc = drv->device_get(&device, 0);
    if (rc != 0) {
        if (debug) fprintf(stderr, "picograd/cuda: cuDeviceGet -> %d\n", rc);
        cached = PG_ERR_UNSUPPORTED;
        return cached;
    }
    rc = drv->ctx_create(&g_ctx, 0, device);
    if (rc != 0) {
        if (debug) fprintf(stderr, "picograd/cuda: cuCtxCreate -> %d\n", rc);
        cached = PG_ERR_UNSUPPORTED;
        return cached;
    }
    rc = drv->module_load_data(&g_module, sgemm_ptx);
    if (rc != 0) {
        if (debug) fprintf(stderr, "picograd/cuda: cuModuleLoadData sgemm -> %d\n", rc);
        cached = PG_ERR_GEMM;
        return cached;
    }
    rc = drv->module_get_function(&g_fn_sgemm, g_module, "pg_sgemm_kernel");
    if (rc != 0) {
        if (debug) fprintf(stderr, "picograd/cuda: cuModuleGetFunction sgemm -> %d\n", rc);
        cached = PG_ERR_GEMM;
        return cached;
    }

    rc = drv->module_load_data(&g_ops_module, ops_ptx);
    if (rc != 0) {
        if (debug) fprintf(stderr, "picograd/cuda: cuModuleLoadData ops -> %d (continuing with sgemm only)\n", rc);
        cached = PG_OK;
        if (debug) fprintf(stderr, "picograd/cuda: init ok (sgemm only)\n");
        return cached;
    }
    int ops_ok = 1;
    rc = drv->module_get_function(&g_fn_map, g_ops_module, "pg_k_map");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_map -> %d\n", rc); ops_ok = 0; }
    rc = drv->module_get_function(&g_fn_bin, g_ops_module, "pg_k_bin");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_bin -> %d\n", rc); ops_ok = 0; }
    rc = drv->module_get_function(&g_fn_accum_scatter, g_ops_module, "pg_k_accum_scatter");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_accum_scatter -> %d\n", rc); ops_ok = 0; }
    rc = drv->module_get_function(&g_fn_sum_axis, g_ops_module, "pg_k_sum_axis");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_sum_axis -> %d\n", rc); ops_ok = 0; }
    rc = drv->module_get_function(&g_fn_softmax, g_ops_module, "pg_k_softmax");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_softmax -> %d\n", rc); ops_ok = 0; }
    rc = drv->module_get_function(&g_fn_copy_strided, g_ops_module, "pg_k_copy_strided");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/cuda: get pg_k_copy_strided -> %d\n", rc); ops_ok = 0; }

    if (ops_ok) {
        cuda_register_gpu();
        if (debug) fprintf(stderr, "picograd/cuda: init ok (sgemm+ops)\n");
    } else {
        if (debug) fprintf(stderr, "picograd/cuda: init ok (sgemm only, ops fallback to CPU)\n");
    }

    cached = PG_OK;
    ok = 1;
    return cached;
}

static void *cuda_malloc(size_t nbytes)
{
    if (cuda_init() != PG_OK)
        return NULL;
    unsigned long long dptr = 0;
    if (pg_cuda_drv_get(NULL)->mem_alloc(&dptr, nbytes ? nbytes : 1) != 0)
        return NULL;
    return (void *)(uintptr_t)dptr;
}

static void cuda_free(void *p)
{
    if (p)
        pg_cuda_drv_get(NULL)->mem_free((unsigned long long)(uintptr_t)p);
}

static pg_status cuda_h2d(void *dst, const void *src, size_t nbytes)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_COPY;
    unsigned long long d = (unsigned long long)(uintptr_t)dst;
    return pg_cuda_drv_get(NULL)->memcpy_h2d(d, src, nbytes) == 0
               ? PG_OK
               : PG_ERR_COPY;
}

static pg_status cuda_d2h(void *dst, const void *src, size_t nbytes)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_COPY;
    unsigned long long s = (unsigned long long)(uintptr_t)src;
    return pg_cuda_drv_get(NULL)->memcpy_d2h(dst, s, nbytes) == 0
               ? PG_OK
               : PG_ERR_COPY;
}

static pg_status cuda_sync(void)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_SYNC;
    return pg_cuda_drv_get(NULL)->ctx_sync() == 0 ? PG_OK : PG_ERR_SYNC;
}

static void cuda_gemm(size_t m, size_t n, size_t k,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float *c, size_t ldc)
{
    if (lda != k || ldb != n || ldc != n) return;
    if (m > 0xffffffffu || n > 0xffffffffu || k > 0xffffffffu) return;

    if (cuda_init() != PG_OK) {
        return;
    }

    unsigned m32 = (unsigned)m, n32 = (unsigned)n, k32 = (unsigned)k;
    void *params[] = {
        &a, &b, &c,
        &m32, &n32, &k32,
    };

    unsigned gx = (n32 + BTILE - 1) / BTILE;
    unsigned gy = (m32 + BTILE - 1) / BTILE;

    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_sgemm, gx, gy, 1, TPT, TPT, 1, 0, NULL, params, NULL);
    assert(rc == 0);
}

static pg_status cuda_gpu_map(float *out, const float *src, size_t n, int op)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    unsigned n32 = (unsigned)n;
    unsigned op32 = (unsigned)op;
    void *params[] = { &out, &src, &n32, &op32 };
    unsigned gx = ((unsigned)n + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_map, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_bin(float *out, const float *a, const float *b,
                               size_t n, int op, const pg_k_bin_args *args)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    unsigned op32 = (unsigned)op;
    void *params[] = { &out, &a, &b, &op32, (void *)args };
    unsigned gx = ((unsigned)n + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_bin, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_accum_scatter(float *dst, const float *src,
                                         float scale, const pg_k_strides *args)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof scale_bits);
    void *params[] = { &dst, &src, &scale_bits, (void *)args };
    unsigned gx = ((unsigned)args->numel + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_accum_scatter, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_sum_axis(float *out, const float *src, float scale,
                                    size_t outer, size_t len, size_t inner,
                                    size_t keepdim_stride)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof scale_bits);
    unsigned o32 = (unsigned)outer, l32 = (unsigned)len, i32 = (unsigned)inner;
    unsigned ks32 = (unsigned)keepdim_stride;
    void *params[] = { &out, &src, &scale_bits, &o32, &l32, &i32, &ks32 };
    unsigned gx = ((unsigned)(outer * inner) + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_sum_axis, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_softmax(float *out, const float *src,
                                   size_t outer, size_t len, size_t inner)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    unsigned o32 = (unsigned)outer, l32 = (unsigned)len, i32 = (unsigned)inner;
    void *params[] = { &out, &src, &o32, &l32, &i32 };
    unsigned gx = ((unsigned)(outer * inner) + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_softmax, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_copy_strided(float *dst, const float *src,
                                        const pg_k_strides *args)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_GEMM;
    void *params[] = { &dst, &src, (void *)args };
    unsigned gx = ((unsigned)args->numel + PG_CUDA_THREADS - 1) / PG_CUDA_THREADS;
    int rc = pg_cuda_drv_get(NULL)->launch_kernel(
        g_fn_copy_strided, gx, 1, 1, PG_CUDA_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status cuda_gpu_fill(void *p, size_t nbytes, float v)
{
    if (cuda_init() != PG_OK)
        return PG_ERR_COPY;
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    unsigned long long d = (unsigned long long)(uintptr_t)p;
    return pg_cuda_drv_get(NULL)->memset_u32(d, bits, nbytes / 4) == 0
               ? PG_OK
               : PG_ERR_COPY;
}

static void cuda_register_gpu(void)
{
    pg_gpu.map          = cuda_gpu_map;
    pg_gpu.bin          = cuda_gpu_bin;
    pg_gpu.accum_scatter = cuda_gpu_accum_scatter;
    pg_gpu.sum_axis     = cuda_gpu_sum_axis;
    pg_gpu.softmax      = cuda_gpu_softmax;
    pg_gpu.copy_strided = cuda_gpu_copy_strided;
    pg_gpu.fill         = cuda_gpu_fill;
    pg_gpu.copy_d2d     = NULL;
}

const pg_backend_ops pg_backend_cuda = {
    .name = "cuda",
    .init = cuda_init,
    .malloc = cuda_malloc,
    .free = cuda_free,
    .copy_h2d = cuda_h2d,
    .copy_d2h = cuda_d2h,
    .sync = cuda_sync,
    .gemm = cuda_gemm,
};
