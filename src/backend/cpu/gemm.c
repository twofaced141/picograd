#include "gemm.h"

#include <stddef.h>
#include <string.h>

extern void sgemm_avx2_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
extern void sgemm_avx512_micro(size_t k,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float *c, size_t ldc,
                               size_t m, size_t n);

#define PG_KC 256
#define PG_MR 8
#define PG_NR_AVX2    8
#define PG_NR_AVX512 16

typedef void (*pg_gemm_micro_fn)(size_t k,
                                 const float *a, size_t lda,
                                 const float *b, size_t ldb,
                                 float *c, size_t ldc,
                                 size_t m, size_t n);

static pg_gemm_micro_fn pg_pick_micro(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f"))
        return sgemm_avx512_micro;
#endif
    return sgemm_avx2_micro;
}

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                 const float *a, size_t lda,
                 const float *b, size_t ldb,
                 float *c, size_t ldc)
{
    pg_gemm_micro_fn micro = pg_pick_micro();
    size_t nr = micro == sgemm_avx512_micro ? PG_NR_AVX512 : PG_NR_AVX2;

    for (size_t i = 0; i < m; i++)
        memset(c + i * ldc, 0, n * sizeof(float));

    for (size_t i = 0; i < m; i += PG_MR) {
        size_t mi = m - i < PG_MR ? m - i : PG_MR;
        for (size_t kk = 0; kk < k; kk += PG_KC) {
            size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
            for (size_t j = 0; j < n; j += nr) {
                size_t nj = n - j < nr ? n - j : nr;
                micro(kl,
                      a + i * lda + kk, lda,
                      b + kk * ldb + j, ldb,
                      c + i * ldc + j, ldc,
                      mi, nj);
            }
        }
    }
}
