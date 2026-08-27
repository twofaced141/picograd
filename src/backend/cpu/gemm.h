#ifndef PICOGRAD_BACKEND_CPU_GEMM_H
#define PICOGRAD_BACKEND_CPU_GEMM_H

#include <stddef.h>

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc);

// fused: C = A*B + bias (bias size n, broadcast rows); act: 0 none, 1 relu, 2 gelu
#define PG_ACT_NONE 0
#define PG_ACT_RELU 1
#define PG_ACT_GELU 2
void pg_cpu_gemm_fused(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc,
                  const float *bias, int act);

#endif
