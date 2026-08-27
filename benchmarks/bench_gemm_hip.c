#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/backend/backend.h"
#include "../src/backend/cpu/gemm.h"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill(float *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)((i * 37 + 11) % 19) - 9.0f) / 8.0f;
}

static int check_correct(size_t s)
{
    float *ha = malloc(s * s * sizeof(float));
    float *hb = malloc(s * s * sizeof(float));
    float *href = malloc(s * s * sizeof(float));
    float *hgot = malloc(s * s * sizeof(float));
    if (!ha || !hb || !href || !hgot)
        return 1;

    fill(ha, s * s);
    fill(hb, s * s);
    pg_cpu_gemm(s, s, s, ha, s, hb, s, href, s);

    float *da = pg_dev_malloc(s * s * sizeof(float));
    float *db = pg_dev_malloc(s * s * sizeof(float));
    float *dc = pg_dev_malloc(s * s * sizeof(float));
    if (!da || !db || !dc)
        return 1;

    if (pg_copy_h2d(da, ha, s * s * sizeof(float)) != PG_OK ||
        pg_copy_h2d(db, hb, s * s * sizeof(float)) != PG_OK)
        return 1;

    pg_gemm(s, s, s, da, s, db, s, dc, s);
    if (pg_copy_d2h(hgot, dc, s * s * sizeof(float)) != PG_OK)
        return 1;

    double maxerr = 0;
    for (size_t i = 0; i < s * s; i++) {
        double e = fabs((double)hgot[i] - (double)href[i]);
        if (e > maxerr)
            maxerr = e;
    }
    printf("  correctness @%zux%zu: max abs err %.4g\n", s, s, maxerr);

    pg_dev_free(da);
    pg_dev_free(db);
    pg_dev_free(dc);
    free(ha);
    free(hb);
    free(href);
    free(hgot);
    return maxerr > 0.05;
}

int main(void)
{
    if (pg_set_device(PG_DEV_HIP) != PG_OK) {
        printf("bench_gemm_hip: hip not available\n");
        return 1;
    }

    if (check_correct(512)) {
        printf("bench_gemm_hip: correctness check FAILED\n");
        return 1;
    }

    printf("%10s %12s %14s\n", "size", "iters", "GFLOP/s");
    const size_t sizes[] = {512, 1024, 2048, 4096};
    const int iters[] = {20, 10, 5, 3};

    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        size_t s = sizes[t];
        int reps = iters[t];

        float *ha = malloc(s * s * sizeof(float));
        float *hb = malloc(s * s * sizeof(float));
        float *dc = pg_dev_malloc(s * s * sizeof(float));
        if (!ha || !hb || !dc)
            return 1;
        fill(ha, s * s);
        fill(hb, s * s);

        float *da = pg_dev_malloc(s * s * sizeof(float));
        float *db = pg_dev_malloc(s * s * sizeof(float));
        if (!da || !db)
            return 1;
        if (pg_copy_h2d(da, ha, s * s * sizeof(float)) != PG_OK ||
            pg_copy_h2d(db, hb, s * s * sizeof(float)) != PG_OK)
            return 1;

        pg_gemm(s, s, s, da, s, db, s, dc, s);
        if (pg_dev_sync() != PG_OK)
            return 1;

        double t0 = now_sec();
        for (int r = 0; r < reps; r++)
            pg_gemm(s, s, s, da, s, db, s, dc, s);
        if (pg_dev_sync() != PG_OK)
            return 1;
        double dt = now_sec() - t0;

        double gflops = (double)reps * 2.0 * (double)s * (double)s *
                        (double)s / dt / 1e9;
        printf("%10zu %12d %14.1f\n", s, reps, gflops);

        pg_dev_free(da);
        pg_dev_free(db);
        pg_dev_free(dc);
        free(ha);
        free(hb);
    }

    return 0;
}
