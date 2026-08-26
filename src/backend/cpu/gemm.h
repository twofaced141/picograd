#ifndef PICOGRAD_BACKEND_CPU_GEMM_H
#define PICOGRAD_BACKEND_CPU_GEMM_H

#include <stddef.h>

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                 const float *a, size_t lda,
                 const float *b, size_t ldb,
                 float *c, size_t ldc);

#endif
