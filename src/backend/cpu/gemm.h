#ifndef PICOGRAD_BACKEND_CPU_GEMM_H
#define PICOGRAD_BACKEND_CPU_GEMM_H

#include <stddef.h>
#include <stdint.h>
#include "../../core/dtype.h"

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc);

// mixed-precision: A/B f16 or bf16 (uint16_t), C f32, ld* in elements
void pg_cpu_hgemm(size_t m, size_t n, size_t k,
                  const uint16_t *a, size_t lda,
                  const uint16_t *b, size_t ldb,
                  float *c, size_t ldc);
void pg_cpu_bgemm(size_t m, size_t n, size_t k,
                  const uint16_t *a, size_t lda,
                  const uint16_t *b, size_t ldb,
                  float *c, size_t ldc);
void pg_cpu_gemm_ex(pg_dtype dtype,
                    size_t m, size_t n, size_t k,
                    const void *a, size_t lda,
                    const void *b, size_t ldb,
                    float *c, size_t ldc);

// fused with bias f32 + activation after f32 accum (for mixed too)
void pg_cpu_gemm_fused_ex(pg_dtype dtype,
                    size_t m, size_t n, size_t k,
                    const void *a, size_t lda,
                    const void *b, size_t ldb,
                    float *c, size_t ldc,
                    const float *bias, int act);

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
