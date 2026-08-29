#include "gemm.h"

#include "../../thread/pool.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
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

static pg_gemm_micro_fn g_cached_micro = NULL;
static int g_cpu_init_done = 0;

static pg_gemm_micro_fn pg_pick_micro(void)
{
    if (g_cached_micro) return g_cached_micro;
#if defined(__x86_64__) || defined(__i386__)
    if (!g_cpu_init_done) {
        __builtin_cpu_init();
        g_cpu_init_done = 1;
    }
    if (__builtin_cpu_supports("avx512f"))
        g_cached_micro = sgemm_avx512_micro;
    else
        g_cached_micro = sgemm_avx2_micro;
#else
    g_cached_micro = sgemm_avx2_micro;
#endif
    return g_cached_micro;
}

typedef struct {
    size_t m, n, k;
    const float *a, *b;
    float *c;
    size_t lda, ldb, ldc;
    pg_gemm_micro_fn micro;
    size_t nr;
} gemm_par_t;

static inline void pack_A_contig(size_t m, size_t k, const float *src, size_t lda, float *dst){
    for(size_t i=0;i<m;i++) memcpy(dst + i*k, src + i*lda, k*sizeof(float));
}
static inline void pack_B_contig(size_t k, size_t n, const float *src, size_t ldb, float *dst){
    for(size_t p=0;p<k;p++) memcpy(dst + p*n, src + p*ldb, n*sizeof(float));
}

static void gemm_par_fn(void *ctx, size_t start, size_t end) {
    gemm_par_t *p = ctx;
    pg_gemm_micro_fn micro = p->micro;
    size_t nr = p->nr;
    size_t m = p->m, n = p->n, k = p->k;
    const float *a = p->a, *b = p->b;
    float *c = p->c;
    size_t lda = p->lda, ldb = p->ldb, ldc = p->ldc;
    // packing buffers (stack, 64B aligned)
    float packA[PG_MR * PG_KC] __attribute__((aligned(64)));
    float packB[PG_KC * PG_NR_AVX512] __attribute__((aligned(64)));
    // start/end are block indices (each block = PG_MR rows)
    for (size_t bi = start; bi < end; bi++) {
        size_t i = bi * PG_MR;
        if (i >= m) break;
        size_t mi = m - i < PG_MR ? m - i : PG_MR;
        // zero this block's C rows
        for (size_t ii = 0; ii < mi; ii++) memset(c + (i+ii)*ldc, 0, n * sizeof(float));
        for (size_t kk = 0; kk < k; kk += PG_KC) {
            size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
            // pack A block once per kk
            pack_A_contig(mi, kl, a + i * lda + kk, lda, packA);
            for (size_t j = 0; j < n; j += nr) {
                size_t nj = n - j < nr ? n - j : nr;
                pack_B_contig(kl, nj, b + kk * ldb + j, ldb, packB);
                micro(kl,
                      packA, kl,
                      packB, nj,
                      c + i * ldc + j, ldc,
                      mi, nj);
            }
        }
    }
}

static void gemm_par_fn_nopack(void *ctx, size_t start, size_t end) {
    gemm_par_t *p = ctx;
    pg_gemm_micro_fn micro = p->micro;
    size_t nr = p->nr;
    size_t m = p->m, n = p->n, k = p->k;
    const float *a = p->a, *b = p->b;
    float *c = p->c;
    size_t lda = p->lda, ldb = p->ldb, ldc = p->ldc;
    for (size_t bi = start; bi < end; bi++) {
        size_t i = bi * PG_MR;
        if (i >= m) break;
        size_t mi = m - i < PG_MR ? m - i : PG_MR;
        for (size_t ii = 0; ii < mi; ii++) memset(c + (i+ii)*ldc, 0, n * sizeof(float));
        for (size_t kk = 0; kk < k; kk += PG_KC) {
            size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
            for (size_t j = 0; j < n; j += nr) {
                size_t nj = n - j < nr ? n - j : nr;
                micro(kl, a + i * lda + kk, lda, b + kk * ldb + j, ldb, c + i * ldc + j, ldc, mi, nj);
            }
        }
    }
}

static void pg_cpu_gemm_serial_packed(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc,
                  pg_gemm_micro_fn micro, size_t nr){
    float packA[PG_MR * PG_KC] __attribute__((aligned(64)));
    float packB[PG_KC * PG_NR_AVX512] __attribute__((aligned(64)));
    for (size_t i = 0; i < m; i++) memset(c + i * ldc, 0, n * sizeof(float));
    for (size_t i = 0; i < m; i += PG_MR) {
        size_t mi = m - i < PG_MR ? m - i : PG_MR;
        for (size_t kk = 0; kk < k; kk += PG_KC) {
            size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
            pack_A_contig(mi, kl, a + i * lda + kk, lda, packA);
            for (size_t j = 0; j < n; j += nr) {
                size_t nj = n - j < nr ? n - j : nr;
                pack_B_contig(kl, nj, b + kk * ldb + j, ldb, packB);
                micro(kl, packA, kl, packB, nj, c + i * ldc + j, ldc, mi, nj);
            }
        }
    }
}

void pg_cpu_gemm(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc)
{
    pg_gemm_micro_fn micro = pg_pick_micro();
    size_t nr = micro == sgemm_avx512_micro ? PG_NR_AVX512 : PG_NR_AVX2;

    size_t total = m * n * k;
    int nthreads = pg_thread_pool_size();
    // packing is kept for non-contiguous views (tensordot) - for contiguous data direct is faster on this uarch
    // enable via env PG_PACK=1 or for very large matrices if needed
    bool use_pack = false;
    const char *env = getenv("PG_PACK");
    if (env && env[0]=='1') use_pack = (total >= (1<<18) && k >= 32 && n >= 16);
    // For small matrices, run serial without pack to avoid overhead
    if (total < (1<<20) || m < 32) {
        for (size_t i = 0; i < m; i++) memset(c + i * ldc, 0, n * sizeof(float));
        for (size_t i = 0; i < m; i += PG_MR) {
            size_t mi = m - i < PG_MR ? m - i : PG_MR;
            for (size_t kk = 0; kk < k; kk += PG_KC) {
                size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
                for (size_t j = 0; j < n; j += nr) {
                    size_t nj = n - j < nr ? n - j : nr;
                    micro(kl, a + i * lda + kk, lda, b + kk * ldb + j, ldb, c + i * ldc + j, ldc, mi, nj);
                }
            }
        }
        return;
    }
    if (nthreads <= 1) {
        if (use_pack) pg_cpu_gemm_serial_packed(m,n,k,a,lda,b,ldb,c,ldc,micro,nr);
        else {
            for (size_t i = 0; i < m; i++) memset(c + i * ldc, 0, n * sizeof(float));
            for (size_t i = 0; i < m; i += PG_MR) {
                size_t mi = m - i < PG_MR ? m - i : PG_MR;
                for (size_t kk = 0; kk < k; kk += PG_KC) {
                    size_t kl = k - kk < PG_KC ? k - kk : PG_KC;
                    for (size_t j = 0; j < n; j += nr) {
                        size_t nj = n - j < nr ? n - j : nr;
                        micro(kl, a + i * lda + kk, lda, b + kk * ldb + j, ldb, c + i * ldc + j, ldc, mi, nj);
                    }
                }
            }
        }
        return;
    }

    // large + multithreaded -> use packed parallel
    size_t nblocks = (m + PG_MR - 1) / PG_MR;
    gemm_par_t ctx={m,n,k,a,b,c,lda,ldb,ldc,micro,nr};
    size_t thresh = 2;
    if (use_pack) pg_parallel_for(nblocks, thresh, gemm_par_fn, &ctx);
    else pg_parallel_for(nblocks, thresh, gemm_par_fn_nopack, &ctx);
}

// ---------- fused gemm with bias + act ----------
static inline float act_apply(float x, int act){
    if(act==PG_ACT_RELU) return x > 0 ? x : 0;
    if(act==PG_ACT_GELU) return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
    return x;
}

typedef struct { size_t m,n,k; const float *a,*b,*bias; float *c; size_t lda,ldb,ldc; int act; pg_gemm_micro_fn micro; size_t nr; } gemm_fused_t;

static void gemm_fused_par(void *ctx, size_t start, size_t end){
    gemm_fused_t *p=ctx;
    pg_gemm_micro_fn micro=p->micro; size_t nr=p->nr;
    size_t m=p->m, n=p->n, k=p->k;
    const float *a=p->a,*b=p->b,*bias=p->bias; float *c=p->c;
    size_t lda=p->lda, ldb=p->ldb, ldc=p->ldc; int act=p->act;
    for(size_t bi=start; bi<end; ++bi){
        size_t i=bi*PG_MR; if(i>=m) break;
        size_t mi=m - i < PG_MR ? m - i : PG_MR;
        // init C with bias (or zero)
        if(bias){
            for(size_t ii=0; ii<mi; ++ii){
                float *crow=c + (i+ii)*ldc;
                for(size_t j=0;j<n;j++) crow[j]=bias[j];
            }
        } else {
            for(size_t ii=0; ii<mi; ++ii) memset(c + (i+ii)*ldc, 0, n*sizeof(float));
        }
        for(size_t kk=0; kk<k; kk+=PG_KC){
            size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
            for(size_t j=0;j<n;j+=nr){
                size_t nj=n - j < nr ? n - j : nr;
                micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
            }
        }
        if(act!=PG_ACT_NONE){
            for(size_t ii=0; ii<mi; ++ii){
                float *crow=c + (i+ii)*ldc;
                if(act==PG_ACT_RELU){
                    for(size_t j=0;j<n;j++) crow[j]= crow[j]>0?crow[j]:0;
                } else {
                    for(size_t j=0;j<n;j++) crow[j]=act_apply(crow[j], act);
                }
            }
        }
    }
}

void pg_cpu_gemm_fused(size_t m, size_t n, size_t k,
                  const float *a, size_t lda,
                  const float *b, size_t ldb,
                  float *c, size_t ldc,
                  const float *bias, int act){
    pg_gemm_micro_fn micro=pg_pick_micro();
    size_t nr=micro==sgemm_avx512_micro?PG_NR_AVX512:PG_NR_AVX2;
    size_t total=m*n*k;
    int nthreads=pg_thread_pool_size();
    if(nthreads<=1 || total < (1<<18) || m < 16){
        for(size_t i=0;i<m;i+=PG_MR){
            size_t mi=m - i < PG_MR ? m - i : PG_MR;
            if(bias){ for(size_t ii=0;ii<mi;ii++){ float*crow=c+(i+ii)*ldc; for(size_t j=0;j<n;j++) crow[j]=bias[j]; } }
            else { for(size_t ii=0;ii<mi;ii++) memset(c+(i+ii)*ldc,0,n*sizeof(float)); }
            for(size_t kk=0;kk<k;kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a+i*lda+kk, lda, b+kk*ldb+j, ldb, c+i*ldc+j, ldc, mi, nj);
                }
            }
            if(act!=PG_ACT_NONE){
                for(size_t ii=0;ii<mi;ii++){
                    float*crow=c+(i+ii)*ldc;
                    if(act==PG_ACT_RELU){
                        for(size_t j=0;j<n;j++) crow[j]=crow[j]>0?crow[j]:0;
                    } else for(size_t j=0;j<n;j++) crow[j]=act_apply(crow[j],act);
                }
            }
        }
        return;
    }
    size_t nblocks=(m+PG_MR-1)/PG_MR;
    gemm_fused_t ctx={m,n,k,a,b,bias,c,lda,ldb,ldc,act,micro,nr};
    pg_parallel_for(nblocks, 2, gemm_fused_par, &ctx);
}
