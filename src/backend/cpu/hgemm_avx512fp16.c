#include <stddef.h>
#include <stdint.h>
#include "../../core/convert.h"
#if defined(__AVX512FP16__)
#include <immintrin.h>
__attribute__((target("avx512fp16,avx512f")))
void hgemm_avx512fp16_micro_c(size_t k, const uint16_t *a, size_t lda,
                             const uint16_t *b, size_t ldb,
                             float *c, size_t ldc, size_t m, size_t n){
    if (m==0 || n==0 || k==0) return;
    if (m > 16 || n > 32) {
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
    // Vectorized AVX512FP16: 16-wide F16 -> F32 convert + FMA
    // n is handled as 16-wide chunks (up to 32 => 2 chunks)
    size_t n0 = n > 16 ? 16 : n;
    size_t n1 = n > 16 ? n - 16 : 0;
    __m512 acc[16][2];
    for(size_t i=0;i<m;i++){
        float *crow = c + i*ldc;
        for(size_t ch=0; ch<2; ch++){
            if(ch==0 && n0>0){
                if(n0==16) acc[i][0]=_mm512_loadu_ps(crow);
                else { __mmask16 mask=(1u<<n0)-1; acc[i][0]=_mm512_maskz_loadu_ps(mask, crow); }
            } else acc[i][0]=_mm512_setzero_ps();
            if(ch==1 && n1>0){
                if(n1==16) acc[i][1]=_mm512_loadu_ps(crow+16);
                else { __mmask16 mask=(1u<<n1)-1; acc[i][1]=_mm512_maskz_loadu_ps(mask, crow+16); }
            } else acc[i][1]=_mm512_setzero_ps();
        }
    }
    for(size_t p=0;p<k;p++){
        __m512 vb0=_mm512_setzero_ps(), vb1=_mm512_setzero_ps();
        if(n0>0){
            __m256i bh = _mm256_loadu_si256((__m256i const*)(b + p*ldb));
            vb0 = _mm512_cvtph_ps(bh);
        }
        if(n1>0){
            __m256i bh1 = _mm256_loadu_si256((__m256i const*)(b + p*ldb + 16));
            vb1 = _mm512_cvtph_ps(bh1);
        }
        for(size_t i=0;i<m;i++){
            float av = pg_f16_to_f32_scalar(a[i*lda + p]);
            __m512 va = _mm512_set1_ps(av);
            if(n0>0) acc[i][0] = _mm512_fmadd_ps(va, vb0, acc[i][0]);
            if(n1>0) acc[i][1] = _mm512_fmadd_ps(va, vb1, acc[i][1]);
        }
    }
    for(size_t i=0;i<m;i++){
        float *crow = c + i*ldc;
        if(n0>0){
            if(n0==16) _mm512_storeu_ps(crow, acc[i][0]);
            else { __mmask16 mask=(1u<<n0)-1; _mm512_mask_storeu_ps(crow, mask, acc[i][0]); }
        }
        if(n1>0){
            if(n1==16) _mm512_storeu_ps(crow+16, acc[i][1]);
            else { __mmask16 mask=(1u<<n1)-1; _mm512_mask_storeu_ps(crow+16, mask, acc[i][1]); }
        }
    }
}
#else
void hgemm_avx512fp16_micro_c(size_t k, const uint16_t *a, size_t lda,
                             const uint16_t *b, size_t ldb,
                             float *c, size_t ldc, size_t m, size_t n){
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i * ldc + j];
            for (size_t p = 0; p < k; p++)
                acc += pg_f16_to_f32_scalar(a[i * lda + p]) * pg_f16_to_f32_scalar(b[p * ldb + j]);
            c[i * ldc + j] = acc;
        }
    }
}
#endif
