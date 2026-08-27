#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/backend/backend.h"
#include "../src/backend/cpu/gemm.h"

static int fails = 0;

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            printf("FAIL:%d %s\n", __LINE__, #c);                      \
            fails++;                                                   \
        }                                                              \
    } while (0)

static void fill(float *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)((i * 37 + 11) % 19) - 9.0f) / 8.0f;
}

static void ref_gemm(size_t m, size_t n, size_t k,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     float *c, size_t ldc)
{
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0;
            for (size_t p = 0; p < k; p++)
                s += (double)a[i * lda + p] * (double)b[p * ldb + j];
            c[i * ldc + j] = (float)s;
        }
}

static void check_result(const char *tag, size_t m, size_t n, size_t k,
                         size_t ldc, const float *got, const float *a,
                         size_t lda, const float *b, size_t ldb)
{
    size_t rc = n + 4;
    float *ref = malloc(m * rc * sizeof(float));
    ref_gemm(m, n, k, a, lda, b, ldb, ref, rc);

    double maxerr = 0;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double e = fabs((double)got[i * ldc + j] - (double)ref[i * rc + j]);
            if (e > maxerr)
                maxerr = e;
        }

    if (maxerr > 1e-2) {
        printf("FAIL %s: max abs err %f\n", tag, maxerr);
        fails++;
    }
    free(ref);
}

static void test_cpu_dispatch(void)
{
    CHECK(pg_set_device(PG_DEV_CPU) == PG_OK);
    CHECK(pg_get_device() == PG_DEV_CPU);
    CHECK(pg_set_device(PG_DEV_METAL) == PG_ERR_UNSUPPORTED);

    enum { M = 13, N = 9, K = 7 };
    float a[M * K], b[K * N], c[M * N];
    fill(a, M * K);
    fill(b, K * N);

    pg_gemm(M, N, K, a, K, b, N, c, N);
    check_result("cpu", M, N, K, N, c, a, K, b, N);

    float v[4] = {1, 2, 3, 4};
    float w[2] = {5, 6};
    float r[2] = {0};
    pg_gemm(2, 1, 2, v, 2, w, 1, r, 1);
    CHECK(r[0] == 17.0f && r[1] == 39.0f);
}

static void test_one_device(pg_devtype dev, const char *name)
{
    if (pg_set_device(dev) != PG_OK) {
        printf("test_backend: %s not available, skipping\n", name);
        return;
    }
    CHECK(pg_get_device() == dev);

    enum { M = 13, N = 9, K = 7, LDA = K, LDB = N, LDC = N };

    float *ha = malloc(M * K * sizeof(float));
    float *hb = malloc(K * N * sizeof(float));
    float *hc = malloc(M * N * sizeof(float));
    fill(ha, M * K);
    fill(hb, K * N);

    float *da = pg_dev_malloc(M * K * sizeof(float));
    float *db = pg_dev_malloc(K * N * sizeof(float));
    float *dc = pg_dev_malloc(M * N * sizeof(float));
    CHECK(da && db && dc);

    CHECK(pg_copy_h2d(da, ha, M * K * sizeof(float)) == PG_OK);
    CHECK(pg_copy_h2d(db, hb, K * N * sizeof(float)) == PG_OK);

    pg_gemm(M, N, K, da, LDA, db, LDB, dc, LDC);
    CHECK(pg_dev_sync() == PG_OK);
    CHECK(pg_copy_d2h(hc, dc, M * N * sizeof(float)) == PG_OK);
    check_result(name, M, N, K, LDC, hc, ha, LDA, hb, LDB);

    pg_dev_free(da);
    pg_dev_free(db);
    pg_dev_free(dc);
    free(ha);
    free(hb);
    free(hc);

    CHECK(pg_set_device(PG_DEV_CPU) == PG_OK);
    CHECK(pg_get_device() == PG_DEV_CPU);
}

static void test_device_roundtrip(void)
{
    test_one_device(PG_DEV_CUDA, "cuda");
    test_one_device(PG_DEV_HIP, "hip");
    test_one_device(PG_DEV_METAL, "metal");
}

int main(void)
{
    test_cpu_dispatch();
    test_device_roundtrip();

    if (fails == 0)
        printf("test_backend: all passed\n");
    else
        printf("test_backend: %d failures\n", fails);
    return fails != 0;
}
