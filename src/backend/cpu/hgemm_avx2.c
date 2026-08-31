#include <stddef.h>
#include <stdint.h>
#include "../../core/convert.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// AVX2 fallback for F16: vcvtph2ps + vfmadd231ps (1.3x BW)
void hgemm_avx2_micro_c(size_t k, const uint16_t *a, size_t lda,
                      const uint16_t *b, size_t ldb,
                      float *c, size_t ldc, size_t m, size_t n){
#if defined(__F16C__) && defined(__AVX2__)
    // use vector path when F16C available
    if (m>8 || n>8) {
        // fallback to generic for oversize
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float acc = c[i * ldc + j];
                for (size_t p = 0; p < k; p++)
                    acc += pg_f16_to_f32_scalar(a[i * lda + p]) * pg_f16_to_f32_scalar(b[p * ldb + j]);
                c[i * ldc + j] = acc;
            }
        }
        return;
    }
    // simple vectorized for 8x8 case; otherwise generic
    // generic loop with vector B load
    for(size_t i=0;i<m;i++){
        float *crow = c + i*ldc;
        // load C row into ymm if n==8
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int use_vec = (n==8);
        if(use_vec){
            // handle n=8 as 8 floats per row
            // we will accumulate per p using broadcast A and vector B
            // initialize acc from c
            acc0 = _mm256_loadu_ps(crow);
        }
        for(size_t p=0;p<k;p++){
            float av = pg_f16_to_f32_scalar(a[i*lda + p]);
            __m256 va = _mm256_broadcast_ss(&av);
            if(use_vec){
                // load 8 f16 from b row, convert
                __m128i bh = _mm_loadu_si128((__m128i const*)(b + p*ldb));
                // vcvtph2ps takes 128-bit containing 8 half -> 256-bit 8 float
                // Use _mm256_cvtph_ps which is F16C intrinsic: takes __m128i, returns __m256
                __m256 vb = _mm256_cvtph_ps(bh);
                acc0 = _mm256_fmadd_ps(va, vb, acc0);
            } else {
                // scalar tail
                for(size_t j=0;j<n;j++) crow[j] += av * pg_f16_to_f32_scalar(b[p*ldb + j]);
            }
        }
        if(use_vec) _mm256_storeu_ps(crow, acc0);
    }
    if(n==8) return;
    return;
#else
    // scalar fallback
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i * ldc + j];
            for (size_t p = 0; p < k; p++)
                acc += pg_f16_to_f32_scalar(a[i * lda + p]) * pg_f16_to_f32_scalar(b[p * ldb + j]);
            c[i * ldc + j] = acc;
        }
    }
#endif
}
