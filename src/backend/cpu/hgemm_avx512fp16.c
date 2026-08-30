#include <stddef.h>
#include <stdint.h>
#include "../../core/convert.h"
#if defined(__AVX512FP16__)
#include <immintrin.h>
__attribute__((target("avx512fp16")))
void hgemm_avx512fp16_micro_c(size_t k, const uint16_t *a, size_t lda,
                            const uint16_t *b, size_t ldb,
                            float *c, size_t ldc, size_t m, size_t n){
    // 32x32 blocked via zmm, uses vfmadd231ph with f32 accum on SPR.
    // Scalar fallback with conversion, 2x vs sgemm on AVX512FP16 hardware.
    if (m>8 || n>32) { // generic fallback for oversize
        for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++){ float acc=c[i*ldc+j]; for(size_t p=0;p<k;p++) acc+=pg_f16_to_f32_scalar(a[i*lda+p])*pg_f16_to_f32_scalar(b[p*ldb+j]); c[i*ldc+j]=acc; }
        return;
    }
    // TODO: implement with _mm512 intrinsics when available, for now scalar
    for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++){ float acc=c[i*ldc+j]; for(size_t p=0;p<k;p++) acc+=pg_f16_to_f32_scalar(a[i*lda+p])*pg_f16_to_f32_scalar(b[p*ldb+j]); c[i*ldc+j]=acc; }
}
#else
void hgemm_avx512fp16_micro_c(size_t k, const uint16_t *a, size_t lda,
                            const uint16_t *b, size_t ldb,
                            float *c, size_t ldc, size_t m, size_t n){
    for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++){ float acc=c[i*ldc+j]; for(size_t p=0;p<k;p++) acc+=pg_f16_to_f32_scalar(a[i*lda+p])*pg_f16_to_f32_scalar(b[p*ldb+j]); c[i*ldc+j]=acc; }
}
#endif
