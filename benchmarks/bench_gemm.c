#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#if defined(PG_ARCH_AARCH64) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC) && defined(__aarch64__))
#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif

#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
#include <mkl.h>
#define PG_BENCH_HAS_MKL 1
#else
#define PG_BENCH_HAS_MKL 0
#endif

#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
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

#if defined(PG_ARCH_AARCH64) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC) && defined(__aarch64__))
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
#endif

#if defined(PG_ARCH_GENERIC) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_AARCH64) && !defined(__x86_64__) && !defined(__i386__) && !defined(__aarch64__))
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#endif

typedef void (*gemm_micro_fn)(size_t k,
                              const float *a, size_t lda,
                              const float *b, size_t ldb,
                              float *c, size_t ldc,
                              size_t m, size_t n);

static gemm_micro_fn g_fn;
static size_t g_nr;

static volatile float sink;

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double bench_ours(size_t k, size_t reps, const float *a, const float *b, float *c)
{
    double t0 = now_sec();
    for (size_t r = 0; r < reps; r++)
        g_fn(k, a, k, b, g_nr, c, g_nr, 8, g_nr);
    return now_sec() - t0;
}

#if PG_BENCH_HAS_MKL
static double bench_mkl(size_t k, size_t reps, const float *a, const float *b, float *c)
{
    double t0 = now_sec();
    for (size_t r = 0; r < reps; r++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    8, (MKL_INT)g_nr, (MKL_INT)k,
                    1.0f, a, (MKL_INT)k, b, (MKL_INT)g_nr, 1.0f, c, (MKL_INT)g_nr);
    return now_sec() - t0;
}
#endif

typedef double (*bench_fn)(size_t, size_t, const float *, const float *, float *);

static double best_time(bench_fn fn, size_t k, const float *a, const float *b, float *c)
{
    size_t reps = 256;
    double t = 0;
    for (;;) {
        t = fn(k, reps, a, b, c);
        if (t > 0.05)
            break;
        reps *= 4;
    }
    for (int i = 0; i < 3; i++) {
        double t2 = fn(k, reps, a, b, c);
        if (t2 < t)
            t = t2;
    }
    sink += c[0] + c[g_nr - 1];
    return t / (double)reps;
}

int main(void)
{
    static const size_t ks[] = { 64, 128, 256, 512, 1024 };

#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
    __builtin_cpu_init();
    int has_avx512 = __builtin_cpu_supports("avx512f") != 0;
    struct {
        const char *name;
        gemm_micro_fn fn;
        size_t nr;
    } kernels[] = {
        { "ours avx2", sgemm_avx2_micro,   8 },
        { "mkl 8x8",   NULL,               8 },
        { "ours avx512", sgemm_avx512_micro, 16 },
        { "mkl 8x16",  NULL,              16 },
    };
    size_t n_cols = has_avx512 ? 4 : 2;
    size_t n_cols_full = sizeof(kernels)/sizeof(kernels[0]);
    int has_sve = 0; (void)has_sve;
    (void)n_cols;
#elif defined(PG_ARCH_AARCH64) || (!defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC) && defined(__aarch64__))
#if defined(__aarch64__) && defined(__linux__)
    int has_sve = 0;
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
    {
        unsigned long hw = getauxval(AT_HWCAP);
#ifdef HWCAP_SVE
        if (hw & HWCAP_SVE) has_sve = 1;
#endif
    }
#endif
#else
    int has_sve = 0;
#if defined(PG_HAVE_SVE)
    has_sve = 1;
#endif
#endif
    struct {
        const char *name;
        gemm_micro_fn fn;
        size_t nr;
    } kernels[] = {
        { "ours neon 8x8", sgemm_neon_micro, 8 },
        { "generic 8x8", sgemm_generic_micro, 8 },
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
        { "ours sve 8x16", sgemm_sve_micro, 16 },
#endif
        { "generic 8x16", sgemm_generic_micro, 16 },
    };
    size_t n_cols_full = sizeof(kernels)/sizeof(kernels[0]);
    // count active columns (skip sve if runtime lacks it)
    size_t n_cols = 0;
    for (size_t i=0;i<n_cols_full;i++) {
        if (!has_sve && strstr(kernels[i].name, "sve")) continue;
        n_cols++;
    }
#else // generic
    struct {
        const char *name;
        gemm_micro_fn fn;
        size_t nr;
    } kernels[] = {
        { "ours generic 8x8", sgemm_generic_micro, 8 },
        { "ours generic 8x16", sgemm_generic_micro, 16 },
    };
    size_t n_cols = sizeof(kernels)/sizeof(kernels[0]);
    size_t n_cols_full = sizeof(kernels)/sizeof(kernels[0]);
    int has_sve = 0; (void)has_sve; (void)n_cols;
#endif

    float *a  = malloc(8 * 1024 * sizeof(float));
    float *b1 = malloc(1024 * 16 * sizeof(float));
    float *b2 = malloc(1024 * 16 * sizeof(float));
    float *c1 = malloc(8 * 16 * sizeof(float));
    float *c2 = malloc(8 * 16 * sizeof(float));

    srand(42);
    for (size_t i = 0; i < 8 * 1024; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < 1024 * 16; i++) { b1[i] = (float)rand() / RAND_MAX - 0.5f; b2[i] = b1[i]; }

    printf("%6s", "K");
    for (size_t col=0; col<n_cols_full; col++) {
        if (!has_sve && strstr(kernels[col].name, "sve")) continue;
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
        if (!has_avx512 && strstr(kernels[col].name, "avx512")) continue;
#endif
        printf(" %12s", kernels[col].name);
    }
    printf("\n");
    for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        size_t k = ks[ki];
        printf("%6zu", k);
        for (size_t col = 0; col < n_cols_full; col++) {
            if (!has_sve && strstr(kernels[col].name, "sve")) continue;
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
            if (!has_avx512 && strstr(kernels[col].name, "avx512")) continue;
#endif
            g_fn = kernels[col].fn;
            g_nr = kernels[col].nr;
            double fl = 2.0 * 8.0 * g_nr * (double)k;
#if PG_BENCH_HAS_MKL
            if (!g_fn) {
                double tm = best_time(bench_mkl, k, a, b1, c2);
                printf(" %11.2f", fl / tm / 1e9);
            } else {
                double to = best_time(bench_ours, k, a, b1, c1);
                printf(" %11.2f", fl / to / 1e9);
            }
#else
            double to = best_time(bench_ours, k, a, b1, c1);
            // for generic duplicate baseline, use same bench_ours (will compare generic vs neon/sve)
            printf(" %11.2f", fl / to / 1e9);
            (void)b2; (void)c2;
#endif
        }
        printf("\n");
    }
    printf("(sink=%g)\n", sink);
    free(a); free(b1); free(b2); free(c1); free(c2);
    return 0;
}
