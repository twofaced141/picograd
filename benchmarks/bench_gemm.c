#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);

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
        sgemm_avx2_micro(k, a, k, b, 8, c, 8, 8, 8);
    return now_sec() - t0;
}

static double bench_mkl(size_t k, size_t reps, const float *a, const float *b, float *c)
{
    double t0 = now_sec();
    for (size_t r = 0; r < reps; r++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    8, 8, (MKL_INT)k, 1.0f, a, (MKL_INT)k, b, 8, 1.0f, c, 8);
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
    sink += c[0] + c[7];
    return t / (double)reps;
}

int main(void)
{
    static const size_t ks[] = { 64, 128, 256, 512, 1024 };
    float *a = malloc(8 * 1024 * sizeof(float));
    float *b1 = malloc(1024 * 8 * sizeof(float));
    float *b2 = malloc(1024 * 8 * sizeof(float));
    float *c1 = malloc(64 * sizeof(float));
    float *c2 = malloc(64 * sizeof(float));

    srand(42);
    for (size_t i = 0; i < 8 * 1024; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < 1024 * 8; i++) { b1[i] = (float)rand() / RAND_MAX - 0.5f; b2[i] = b1[i]; }

    printf("%6s %12s %12s\n", "K", "ours GF/s", "mkl GF/s");
    for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        size_t k = ks[ki];
        double to = best_time(bench_ours, k, a, b1, c1);
        double tm = best_time(bench_mkl, k, a, b2, c2);
        double fl = 2.0 * 8.0 * 8.0 * (double)k;
        printf("%6zu %11.2f %11.2f\n", k, fl / to / 1e9, fl / tm / 1e9);
    }
    printf("(sink=%g)\n", sink);
    free(a); free(b1); free(b2); free(c1); free(c2);
    return 0;
}
