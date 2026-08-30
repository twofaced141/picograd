#include <stddef.h>

void sgemm_generic_micro(size_t k,
                         const float *a, size_t lda,
                         const float *b, size_t ldb,
                         float *c, size_t ldc,
                         size_t m, size_t n)
{
    if (m==0 || n==0) return;
    if (k==0) return;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i*ldc + j];
            const float *arow = a + i*lda;
            for (size_t p = 0; p < k; p++) {
                acc += arow[p] * b[p*ldb + j];
            }
            c[i*ldc + j] = acc;
        }
    }
}
