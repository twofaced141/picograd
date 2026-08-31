#include <stddef.h>
#include <stdint.h>
#include "../../core/convert.h"
#if defined(__AVX512BF16__)
#include <immintrin.h>
__attribute__((target("avx512bf16")))
void bgemm_avx512bf16_micro_c(size_t k, const uint16_t *a, size_t lda,
                            const uint16_t *b, size_t ldb,
                            float *c, size_t ldc, size_t m, size_t n){
    // Use vcvtneps2bf16 + vdpbf16ps zmm - 2x vs sgemm
    // For now scalar fallback
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i * ldc + j];
            for (size_t p = 0; p < k; p++)
                acc += pg_bf16_to_f32_scalar(a[i * lda + p]) * pg_bf16_to_f32_scalar(b[p * ldb + j]);
            c[i * ldc + j] = acc;
        }
    }
}
#else
void bgemm_avx512bf16_micro_c(size_t k, const uint16_t *a, size_t lda,
                            const uint16_t *b, size_t ldb,
                            float *c, size_t ldc, size_t m, size_t n){
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i * ldc + j];
            for (size_t p = 0; p < k; p++)
                acc += pg_bf16_to_f32_scalar(a[i * lda + p]) * pg_bf16_to_f32_scalar(b[p * ldb + j]);
            c[i * ldc + j] = acc;
        }
    }
}
#endif
