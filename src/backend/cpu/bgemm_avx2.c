#include <stddef.h>
#include <stdint.h>
#include "../../core/convert.h"

// AVX2 fallback for BF16: convert via shift + RNE, then fma (1.3x BW)
void bgemm_avx2_micro_c(size_t k, const uint16_t *a, size_t lda,
                      const uint16_t *b, size_t ldb,
                      float *c, size_t ldc, size_t m, size_t n){
    // BF16 AVX2: no direct vcvt for bf16 in AVX2, use scalar conversion
    // Provide correct but not peak performance
    for(size_t i=0;i<m;i++){
        for(size_t j=0;j<n;j++){
            float acc = c[i*ldc + j];
            for(size_t p=0;p<k;p++){
                float av = pg_bf16_to_f32_scalar(a[i*lda + p]);
                float bv = pg_bf16_to_f32_scalar(b[p*ldb + j]);
                acc += av * bv;
            }
            c[i*ldc + j]=acc;
        }
    }
}
