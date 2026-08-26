#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);

#define GUARD 16
#define TOL   1e-3f

static unsigned long long rng_state = 0x9e3779b97f4a7c15ULL;

static float frand(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((long)(rng_state >> 36) % 2001 - 1000) / 1000.0f;
}

static void ref_gemm(size_t m, size_t n, size_t k,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     const float *c0, size_t ldc0,
                     double *ref, size_t ldr)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = (double)c0[i * ldc0 + j];
            for (size_t p = 0; p < k; p++)
                s += (double)a[i * lda + p] * (double)b[p * ldb + j];
            ref[i * ldr + j] = s;
        }
    }
}

static int g_details;

static int check_case(size_t m, size_t n, size_t k, int slack)
{
    size_t lda = k ? k + slack : 1;
    size_t ldb = n + slack;
    size_t ldc = n + slack;

    float *a   = malloc(m * lda * sizeof(float));
    float *b   = malloc(k * ldb * sizeof(float));
    float *c0  = malloc(m * ldc * sizeof(float));
    double *ref = malloc(m * ldc * sizeof(double));
    unsigned char *buf = malloc((m * ldc + 2 * GUARD) * sizeof(float));
    float *c = (float *)buf + GUARD;

    for (size_t i = 0; i < m; i++)
        for (size_t p = 0; p < k; p++)
            a[i * lda + p] = frand();
    for (size_t p = 0; p < k; p++)
        for (size_t j = 0; j < ldb; j++)
            b[p * ldb + j] = frand();
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < ldc; j++)
            c0[i * ldc + j] = frand();

    ref_gemm(m, n, k, a, lda, b, ldb, c0, ldc, ref, ldc);

    memset(buf, 0xAB, (m * ldc + 2 * GUARD) * sizeof(float));
    for (size_t i = 0; i < m; i++)
        memcpy(c + i * ldc, c0 + i * ldc, n * sizeof(float));

    sgemm_avx2_micro(k, a, lda, b, ldb, c, ldc, m, n);

    int fail = 0;
    for (size_t i = 0; i < m && !fail; i++) {
        for (size_t j = 0; j < n && !fail; j++) {
            float g = c[i * ldc + j];
            double r = ref[i * ldc + j];
            if (fabsf(g - (float)r) > TOL * fmaxf(1.0f, fabsf((float)r)))
                fail = 1;
        }
    }
    size_t guard_bytes = GUARD * sizeof(float);
    for (size_t i = 0; i < guard_bytes && !fail; i++)
        if (((unsigned char *)buf)[guard_bytes + m * ldc * sizeof(float) + i] != 0xAB)
            fail = 2;
    for (size_t i = 0; i < guard_bytes && !fail; i++)
        if (((unsigned char *)buf)[i] != 0xAB)
            fail = 2;

    if (fail && g_details < 5) {
        g_details++;
        printf("FAIL m=%zu n=%zu k=%zu slack=%d (%s)\n", m, n, k, slack,
               fail == 2 ? "canary overwritten" : "value mismatch");
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++)
                printf("  got=%+.6f ref=%+.6f\n", c[i * ldc + j], ref[i * ldc + j]);
            if (i >= 2) break;
        }
    }

    free(a); free(b); free(c0); free(ref); free(buf);
    return fail != 0;
}

int main(void)
{
    static const size_t ks[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 33 };
    size_t total = 0, fails = 0;

    for (int slack = 0; slack <= 1; slack++)
        for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++)
            for (size_t m = 1; m <= 8; m++)
                for (size_t n = 1; n <= 8; n++) {
                    total++;
                    fails += check_case(m, n, ks[ki], slack);
                }

    printf("%zu/%zu cases passed\n", total - fails, total);
    return fails != 0;
}
