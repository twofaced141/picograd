#include "gemm.h"

#include <stddef.h>
#include <string.h>

extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);

#define PG_KC 256

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                 const float *a, size_t lda,
                 const float *b, size_t ldb,
                 float *c, size_t ldc)
{
    for (size_t i = 0; i < m; i++)
        memset(c + i * ldc, 0, n * sizeof(float));

    for (size_t i = 0; i < m; i += 8) {
        size_t mi = m - i < 8 ? m - i : 8;
        for (size_t kk = 0; kk < k; kk += PG_KC) {
            size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
            for (size_t j = 0; j < n; j += 8) {
                size_t nj = n - j < 8 ? n - j : 8;
                sgemm_avx2_micro(kl,
                                 a + i * lda + kk, lda,
                                 b + kk * ldb + j, ldb,
                                 c + i * ldc + j, ldc,
                                 mi, nj);
            }
        }
    }
}
