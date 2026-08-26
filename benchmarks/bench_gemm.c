#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

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

static double bench_mkl(size_t k, size_t reps, const float *a, const float *b, float *c)
{
    double t0 = now_sec();
    for (size_t r = 0; r < reps; r++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    8, (MKL_INT)g_nr, (MKL_INT)k,
                    1.0f, a, (MKL_INT)k, b, (MKL_INT)g_nr, 1.0f, c, (MKL_INT)g_nr);
    return now_sec() - t0;
}

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
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    int has_avx512 = __builtin_cpu_supports("avx512f") != 0;
#else
    int has_avx512 = 0;
#endif
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

    float *a  = malloc(8 * 1024 * sizeof(float));
    float *b1 = malloc(1024 * 16 * sizeof(float));
    float *b2 = malloc(1024 * 16 * sizeof(float));
    float *c1 = malloc(8 * 16 * sizeof(float));
    float *c2 = malloc(8 * 16 * sizeof(float));

    srand(42);
    for (size_t i = 0; i < 8 * 1024; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < 1024 * 16; i++) { b1[i] = (float)rand() / RAND_MAX - 0.5f; b2[i] = b1[i]; }

    printf("%6s %12s %12s", "K", kernels[0].name, kernels[1].name);
    if (has_avx512)
        printf(" %12s %12s", kernels[2].name, kernels[3].name);
    printf("\n");
    for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        size_t k = ks[ki];
        printf("%6zu", k);
        for (size_t col = 0; col < n_cols; col++) {
            g_fn = kernels[col].fn;
            g_nr = kernels[col].nr;
            double fl = 2.0 * 8.0 * g_nr * (double)k;
            if (!g_fn) {
                double tm = best_time(bench_mkl, k, a, b1, c2);
                printf(" %11.2f", fl / tm / 1e9);
            } else {
                double to = best_time(bench_ours, k, a, b1, c1);
                printf(" %11.2f", fl / to / 1e9);
            }
        }
        printf("\n");
    }
    printf("(sink=%g)\n", sink);
    free(a); free(b1); free(b2); free(c1); free(c2);
    return 0;
}
