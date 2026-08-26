#include "../backend_i.h"
#include "driver.h"

#include "sgemm_ptx.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#define BTILE 64
#define TPT 16

static void *g_ctx;
static void *g_module;
static void *g_fn_sgemm;

static pg_status cuda_init(void)
{
    static pg_status cached = PG_ERR_UNSUPPORTED;
    static int done = 0;

    if (done)
        return cached;
    done = 1;

    pg_status err = PG_OK;
    const pg_cuda_drv *drv = pg_cuda_drv_get(&err);
    if (!drv)
        return cached = err;

    int device;
    if (drv->device_get(&device, 0) != 0)
        return cached = PG_ERR_UNSUPPORTED;
    if (drv->ctx_create(&g_ctx, 0, device) != 0)
        return cached = PG_ERR_UNSUPPORTED;
    if (drv->module_load_data(&g_module, sgemm_ptx) != 0)
        return cached = PG_ERR_GEMM;
    if (drv->module_get_function(&g_fn_sgemm, g_module, "pg_sgemm_kernel") != 0)
        return cached = PG_ERR_GEMM;

    cached = PG_OK;
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
    assert(lda == k && ldb == n && ldc == n);
    assert(m <= 0xffffffffu && n <= 0xffffffffu && k <= 0xffffffffu);

    if (cuda_init() != PG_OK) {
        assert(!"cuda backend not initialized");
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
