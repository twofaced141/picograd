#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__aarch64__) && defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(PG_ARCH_X86_64)
extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
extern void sgemm_avx512_micro(size_t k,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float *c, size_t ldc,
                               size_t m, size_t n);
#elif defined(PG_ARCH_AARCH64)
extern void sgemm_neon_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
extern void sgemm_sve_micro(size_t k,
                            const float *a, size_t lda,
                            const float *b, size_t ldb,
                            float *c, size_t ldc,
                            size_t m, size_t n);
#endif
#elif defined(PG_ARCH_GENERIC)
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#else
// auto-detect fallback for host builds without PG_ARCH
#if defined(__x86_64__) || defined(__i386__)
extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
extern void sgemm_avx512_micro(size_t k,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float *c, size_t ldc,
                               size_t m, size_t n);
#endif
#if defined(__aarch64__)
extern void sgemm_neon_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
extern void sgemm_sve_micro(size_t k,
                            const float *a, size_t lda,
                            const float *b, size_t ldb,
                            float *c, size_t ldc,
                            size_t m, size_t n);
#endif
#elif !defined(__x86_64__) && !defined(__i386__)
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#endif
#endif

typedef void (*gemm_micro_fn)(size_t k,
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
static gemm_micro_fn g_micro;
static const char *g_micro_name = "";

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

    g_micro(k, a, lda, b, ldb, c, ldc, m, n);

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
        printf("FAIL %s m=%zu n=%zu k=%zu slack=%d (%s)\n",
               g_micro_name,
               m, n, k, slack,
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

#if defined(__aarch64__) || defined(PG_ARCH_AARCH64)
static int has_sve_runtime(void){
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
#ifdef HWCAP_SVE
    return (hw & HWCAP_SVE) != 0;
#else
    return 0;
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 12
    return __builtin_cpu_supports("sve") != 0;
#else
    return 0;
#endif
#endif
#else
    return 0;
#endif
}
#endif

int main(void)
{
    static const size_t ks[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 33 };
    struct {
        const char *name;
        gemm_micro_fn fn;
        size_t nr;
        int skip; // set to 1 to skip
    } kernels[] = {
#if defined(PG_ARCH_X86_64)
        { "avx2",   sgemm_avx2_micro,   8, 0 },
        { "avx512", sgemm_avx512_micro, 16, 0 },
#elif defined(PG_ARCH_AARCH64)
        { "neon",    sgemm_neon_micro,    8, 0 },
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
        { "sve",     sgemm_sve_micro,    16, 0 },
#endif
        { "generic", sgemm_generic_micro, 8, 0 },
#elif defined(PG_ARCH_GENERIC)
        { "generic", sgemm_generic_micro, 8, 0 },
#else
// auto-detect host
#if defined(__x86_64__) || defined(__i386__)
        { "avx2",   sgemm_avx2_micro,   8, 0 },
        { "avx512", sgemm_avx512_micro, 16, 0 },
#endif
#if defined(__aarch64__)
        { "neon",    sgemm_neon_micro,    8, 0 },
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
        { "sve",     sgemm_sve_micro,    16, 0 },
#endif
        { "generic", sgemm_generic_micro, 8, 0 },
#elif !defined(__x86_64__) && !defined(__i386__)
        { "generic", sgemm_generic_micro, 8, 0 },
#endif
#endif
    };
    size_t n_kernels = sizeof(kernels) / sizeof(kernels[0]);
    if (n_kernels==0){
        printf("test_gemm: no kernels for this arch\n");
        return 0;
    }
    int has_avx512 = 0;
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
    __builtin_cpu_init();
    has_avx512 = __builtin_cpu_supports("avx512f") != 0;
#endif
#if defined(PG_ARCH_AARCH64) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC) && defined(__aarch64__))
    int has_sve = has_sve_runtime();
#endif

    size_t total = 0, fails = 0;

    for (size_t kn = 0; kn < n_kernels; kn++) {
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
        if (strcmp(kernels[kn].name,"avx512")==0 && !has_avx512) {
            printf("test_gemm: skipping avx512 (not supported by cpu)\n");
            continue;
        }
#endif
#if defined(PG_ARCH_AARCH64) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC) && defined(__aarch64__))
        if (strcmp(kernels[kn].name,"sve")==0 && !has_sve) {
            printf("test_gemm: skipping sve (not supported by cpu)\n");
            continue;
        }
#endif
        if (kernels[kn].skip) continue;
        g_micro = kernels[kn].fn;
        g_micro_name = kernels[kn].name;
        size_t cur_fails_before = fails;
        for (int slack = 0; slack <= 1; slack++)
            for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++)
                for (size_t m = 1; m <= 8; m++)
                    for (size_t n = 1; n <= kernels[kn].nr; n++) {
                        total++;
                        fails += check_case(m, n, ks[ki], slack);
                    }
        printf("test_gemm[%s]: %s\n", kernels[kn].name,
               fails == cur_fails_before ? "all cases passed" : "FAILURES");
    }

    printf("test_gemm: %zu/%zu cases passed\n", total - fails, total);
    return fails != 0;
}
