#if defined(__aarch64__) && defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif
#include "gemm.h"

#include "../../core/convert.h"
#include "../../thread/pool.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if (defined(__aarch64__) || defined(PG_ARCH_AARCH64)) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(PG_ARCH_X86_64)
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
#elif defined(PG_ARCH_AARCH64)
extern void sgemm_neon_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
extern void sgemm_sve_micro(size_t k,
                            const float *a, size_t lda,
                            const float *b, size_t ldb,
                            float *c, size_t ldc,
                            size_t m, size_t n);
#endif
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#elif defined(PG_ARCH_GENERIC)
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#else
#if defined(__x86_64__) || defined(__i386__)
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
#endif
#if defined(__aarch64__)
extern void sgemm_neon_micro(size_t k,
                             const float *a, size_t lda,
                             const float *b, size_t ldb,
                             float *c, size_t ldc,
                             size_t m, size_t n);
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
extern void sgemm_sve_micro(size_t k,
                            const float *a, size_t lda,
                            const float *b, size_t ldb,
                            float *c, size_t ldc,
                            size_t m, size_t n);
#endif
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#elif !defined(__x86_64__) && !defined(__i386__)
extern void sgemm_generic_micro(size_t k,
                                const float *a, size_t lda,
                                const float *b, size_t ldb,
                                float *c, size_t ldc,
                                size_t m, size_t n);
#endif
#endif

// mixed micro prototypes (to be implemented in asm or C)
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
// optional AVX512FP16/BF16 microkernels (may be absent -> fallback)
extern void hgemm_avx512fp16_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n) __attribute__((weak));
extern void bgemm_avx512bf16_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n) __attribute__((weak));
extern void hgemm_avx2_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n) __attribute__((weak));
extern void bgemm_avx2_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n) __attribute__((weak));
#endif

#define PG_KC 256
#define PG_MC 384
#define PG_NC 4096
#define PG_MR 8
#define PG_NR_AVX2    8
#define PG_NR_AVX512 16
#define PG_NR_NEON    8
#define PG_NR_SVE    16
#define PG_NR_GENERIC 8
#define PG_NR_MAX    16

// legacy sgemm micro typedef
typedef void (*pg_gemm_micro_fn)(size_t k,
                                 const float *a, size_t lda,
                                 const float *b, size_t ldb,
                                 float *c, size_t ldc,
                                 size_t m, size_t n);
typedef void (*pg_hgemm_micro_fn)(size_t k,
                                  const uint16_t *a, size_t lda,
                                  const uint16_t *b, size_t ldb,
                                  float *c, size_t ldc,
                                  size_t m, size_t n);
typedef void (*pg_bgemm_micro_fn)(size_t k,
                                  const uint16_t *a, size_t lda,
                                  const uint16_t *b, size_t ldb,
                                  float *c, size_t ldc,
                                  size_t m, size_t n);

static pg_gemm_micro_fn g_cached_micro = NULL;
static size_t g_cached_nr = 0;
static pg_hgemm_micro_fn g_cached_hmicro = NULL;
static pg_bgemm_micro_fn g_cached_bmicro = NULL;
static size_t g_cached_hnr = 0;
static size_t g_cached_bnr = 0;
static int g_cpu_init_done = 0;

// generic fallback micros for mixed (scalar conversion)
static void hgemm_generic_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    if(m==0||n==0||k==0) return;
    for(size_t i=0;i<m;i++){
        for(size_t j=0;j<n;j++){
            float acc = c[i*ldc + j];
            for(size_t p=0;p<k;p++){
                float av = pg_f16_to_f32_scalar(a[i*lda + p]);
                float bv = pg_f16_to_f32_scalar(b[p*ldb + j]);
                acc += av * bv;
            }
            c[i*ldc + j]=acc;
        }
    }
}
static void bgemm_generic_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    if(m==0||n==0||k==0) return;
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
// AVX2 conversion micro: convert f16/bf16 on fly using vcvtph2ps / bf16 via scalar
static void hgemm_avx2_cvt_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    hgemm_generic_micro(k,a,lda,b,ldb,c,ldc,m,n);
}
static void bgemm_avx2_cvt_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    bgemm_generic_micro(k,a,lda,b,ldb,c,ldc,m,n);
}

static pg_gemm_micro_fn pg_pick_micro(void)
{
    if (g_cached_micro) return g_cached_micro;
#if defined(PG_ARCH_X86_64)
    if (!g_cpu_init_done) { __builtin_cpu_init(); g_cpu_init_done = 1; }
    if (__builtin_cpu_supports("avx512f")) {
        g_cached_micro = sgemm_avx512_micro;
        g_cached_nr = PG_NR_AVX512;
    } else {
        g_cached_micro = sgemm_avx2_micro;
        g_cached_nr = PG_NR_AVX2;
    }
#elif defined(PG_ARCH_AARCH64)
    int has_sve = 0;
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
#ifdef HWCAP_SVE
    if (hw & HWCAP_SVE) has_sve = 1;
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 12
    if (__builtin_cpu_supports("sve")) has_sve = 1;
#endif
#endif
    if (has_sve) {
        g_cached_micro = sgemm_sve_micro;
        g_cached_nr = PG_NR_SVE;
        return g_cached_micro;
    }
#endif
    g_cached_micro = sgemm_neon_micro;
    g_cached_nr = PG_NR_NEON;
#elif defined(PG_ARCH_GENERIC)
    g_cached_micro = sgemm_generic_micro;
    g_cached_nr = PG_NR_GENERIC;
#else
#if defined(__x86_64__) || defined(__i386__)
    if (!g_cpu_init_done) { __builtin_cpu_init(); g_cpu_init_done = 1; }
    if (__builtin_cpu_supports("avx512f")) {
        g_cached_micro = sgemm_avx512_micro;
        g_cached_nr = PG_NR_AVX512;
    } else {
        g_cached_micro = sgemm_avx2_micro;
        g_cached_nr = PG_NR_AVX2;
    }
#elif defined(__aarch64__)
    int has_sve = 0;
#if defined(PG_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
#ifdef HWCAP_SVE
    if (hw & HWCAP_SVE) has_sve = 1;
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 12
    if (__builtin_cpu_supports("sve")) has_sve = 1;
#endif
#endif
    if (has_sve) {
        g_cached_micro = sgemm_sve_micro;
        g_cached_nr = PG_NR_SVE;
        return g_cached_micro;
    }
#endif
    g_cached_micro = sgemm_neon_micro;
    g_cached_nr = PG_NR_NEON;
#else
    g_cached_micro = sgemm_generic_micro;
    g_cached_nr = PG_NR_GENERIC;
#endif
#endif
    return g_cached_micro;
}
static inline size_t pg_pick_nr(void){
    if (!g_cached_micro) pg_pick_micro();
    return g_cached_nr;
}

// ---- mixed pickers ----
static pg_hgemm_micro_fn pg_pick_hmicro(size_t *out_nr){
    if (g_cached_hmicro) { *out_nr=g_cached_hnr; return g_cached_hmicro; }
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
    if (!g_cpu_init_done) { __builtin_cpu_init(); g_cpu_init_done=1; }
    // priority: AMX > AVX512FP16 > AVX2 cvt > generic
    // AMX detection requires amx-tile and amx-bf16 support; we currently provide no AMX micro, so skip
#ifdef __AMX_BF16__
    // placeholder for amx_bf16 check – would require tile config
#endif
    if (__builtin_cpu_supports("avx512fp16")) {
        if (hgemm_avx512fp16_micro) { g_cached_hmicro = hgemm_avx512fp16_micro; g_cached_hnr=32; *out_nr=32; return g_cached_hmicro; }
    }
    // AVX2 fallback
    if (hgemm_avx2_micro) { g_cached_hmicro = hgemm_avx2_micro; g_cached_hnr=PG_NR_AVX2; *out_nr=PG_NR_AVX2; return g_cached_hmicro; }
    // our internal AVX2 cvt wrapper (still generic speed)
    g_cached_hmicro = hgemm_avx2_cvt_micro; g_cached_hnr=PG_NR_AVX2; *out_nr=PG_NR_AVX2; return g_cached_hmicro;
#elif defined(PG_ARCH_AARCH64) || (defined(__aarch64__) && !defined(PG_ARCH_X86_64) && !defined(PG_ARCH_GENERIC))
    // ARM NEON fp16 -> future, fallback generic
    g_cached_hmicro = hgemm_generic_micro; g_cached_hnr=PG_NR_NEON; *out_nr=PG_NR_NEON; return g_cached_hmicro;
#else
    g_cached_hmicro = hgemm_generic_micro; g_cached_hnr=PG_NR_GENERIC; *out_nr=PG_NR_GENERIC; return g_cached_hmicro;
#endif
    g_cached_hmicro = hgemm_generic_micro; g_cached_hnr=PG_NR_GENERIC; *out_nr=PG_NR_GENERIC; return g_cached_hmicro;
}
static pg_bgemm_micro_fn pg_pick_bmicro(size_t *out_nr){
    if (g_cached_bmicro) { *out_nr=g_cached_bnr; return g_cached_bmicro; }
#if defined(PG_ARCH_X86_64) || (!defined(PG_ARCH_AARCH64) && !defined(PG_ARCH_GENERIC) && (defined(__x86_64__)||defined(__i386__)))
    if (!g_cpu_init_done) { __builtin_cpu_init(); g_cpu_init_done=1; }
    if (__builtin_cpu_supports("avx512bf16")) {
        if (bgemm_avx512bf16_micro) { g_cached_bmicro = bgemm_avx512bf16_micro; g_cached_bnr=32; *out_nr=32; return g_cached_bmicro; }
    }
    // AMX-BF16 would be priority if available (clang lacks builtin strings)
#if !defined(__clang__)
    if (__builtin_cpu_supports("amx-bf16") && __builtin_cpu_supports("amx-tile")) {
        // placeholder: if amx micro available would be selected
    }
#endif
    if (bgemm_avx2_micro) { g_cached_bmicro = bgemm_avx2_micro; g_cached_bnr=PG_NR_AVX2; *out_nr=PG_NR_AVX2; return g_cached_bmicro; }
    g_cached_bmicro = bgemm_avx2_cvt_micro; g_cached_bnr=PG_NR_AVX2; *out_nr=PG_NR_AVX2; return g_cached_bmicro;
#elif defined(PG_ARCH_AARCH64)
    g_cached_bmicro = bgemm_generic_micro; g_cached_bnr=PG_NR_NEON; *out_nr=PG_NR_NEON; return g_cached_bmicro;
#else
    g_cached_bmicro = bgemm_generic_micro; g_cached_bnr=PG_NR_GENERIC; *out_nr=PG_NR_GENERIC; return g_cached_bmicro;
#endif
    g_cached_bmicro = bgemm_generic_micro; g_cached_bnr=PG_NR_GENERIC; *out_nr=PG_NR_GENERIC; return g_cached_bmicro;
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
static inline void pack_A_h_contig(size_t m, size_t k, const uint16_t *src, size_t lda, uint16_t *dst){
    for(size_t i=0;i<m;i++) memcpy(dst + i*k, src + i*lda, k*sizeof(uint16_t));
}
static inline void pack_B_h_contig(size_t k, size_t n, const uint16_t *src, size_t ldb, uint16_t *dst){
    for(size_t p=0;p<k;p++) memcpy(dst + p*n, src + p*ldb, n*sizeof(uint16_t));
}

static void gemm_par_fn(void *ctx, size_t start, size_t end) {
    gemm_par_t *p = ctx;
    pg_gemm_micro_fn micro = p->micro;
    size_t nr = p->nr;
    size_t m = p->m, n = p->n, k = p->k;
    const float *a = p->a, *b = p->b;
    float *c = p->c;
    size_t lda = p->lda, ldb = p->ldb, ldc = p->ldc;
    float packA[PG_MR * PG_KC] __attribute__((aligned(64)));
    float packB[PG_KC * PG_NR_MAX] __attribute__((aligned(64)));
    for (size_t bi = start; bi < end; bi++) {
        size_t i = bi * PG_MR;
        if (i >= m) break;
        size_t mi = m - i < PG_MR ? m - i : PG_MR;
        for (size_t ii = 0; ii < mi; ii++) memset(c + (i+ii)*ldc, 0, n * sizeof(float));
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
    float packB[PG_KC * PG_NR_MAX] __attribute__((aligned(64)));
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
    size_t nr = pg_pick_nr();
    size_t total = m * n * k;
    int nthreads = pg_thread_pool_size();
    bool use_pack = false;
    const char *env = getenv("PG_PACK");
    if (env && env[0]=='1') use_pack = (total >= (1<<18) && k >= 32 && n >= 16);
    // GotoBLAS: for large matrices packing gives better TLB + prefetch
    // auto-enable packing for standard MC/NC/KC when large
    if (!use_pack && total >= (size_t)PG_MC*PG_NC*PG_KC/4) use_pack = true;

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
    size_t nblocks = (m + PG_MR - 1) / PG_MR;
    gemm_par_t ctx={m,n,k,a,b,c,lda,ldb,ldc,micro,nr};
    size_t thresh = 2;
    if (use_pack) pg_parallel_for(nblocks, thresh, gemm_par_fn, &ctx);
    else pg_parallel_for(nblocks, thresh, gemm_par_fn_nopack, &ctx);
}

// ---------- mixed precision core ----------
typedef struct {
    size_t m,n,k;
    const uint16_t *a,*b;
    float *c;
    size_t lda,ldb,ldc;
    pg_hgemm_micro_fn micro;
    size_t nr;
} hgemm_par_t;
typedef struct {
    size_t m,n,k;
    const uint16_t *a,*b;
    float *c;
    size_t lda,ldb,ldc;
    pg_bgemm_micro_fn micro;
    size_t nr;
} bgemm_par_t;

static void hgemm_par_fn(void *ctx, size_t start, size_t end){
    hgemm_par_t *p=ctx;
    pg_hgemm_micro_fn micro=p->micro; size_t nr=p->nr;
    size_t m=p->m,n=p->n,k=p->k;
    const uint16_t *a=p->a,*b=p->b; float *c=p->c;
    size_t lda=p->lda, ldb=p->ldb, ldc=p->ldc;
    uint16_t packA[PG_MR * PG_KC] __attribute__((aligned(64)));
    uint16_t packB[PG_KC * PG_NR_MAX*2] __attribute__((aligned(64))); // max 32
    for(size_t bi=start;bi<end;bi++){
        size_t i=bi*PG_MR; if(i>=m) break;
        size_t mi=m - i < PG_MR ? m - i : PG_MR;
        for(size_t ii=0;ii<mi;ii++) memset(c + (i+ii)*ldc, 0, n*sizeof(float));
        for(size_t kk=0;kk<k;kk+=PG_KC){
            size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
            pack_A_h_contig(mi, kl, a + i*lda + kk, lda, packA);
            for(size_t j=0;j<n;j+=nr){
                size_t nj=n - j < nr ? n - j : nr;
                pack_B_h_contig(kl, nj, b + kk*ldb + j, ldb, packB);
                micro(kl, packA, kl, packB, nj, c + i*ldc + j, ldc, mi, nj);
            }
        }
    }
}
static void bgemm_par_fn(void *ctx, size_t start, size_t end){
    bgemm_par_t *p=ctx;
    pg_bgemm_micro_fn micro=p->micro; size_t nr=p->nr;
    size_t m=p->m,n=p->n,k=p->k;
    const uint16_t *a=p->a,*b=p->b; float *c=p->c;
    size_t lda=p->lda, ldb=p->ldb, ldc=p->ldc;
    uint16_t packA[PG_MR * PG_KC] __attribute__((aligned(64)));
    uint16_t packB[PG_KC * PG_NR_MAX*2] __attribute__((aligned(64)));
    for(size_t bi=start;bi<end;bi++){
        size_t i=bi*PG_MR; if(i>=m) break;
        size_t mi=m - i < PG_MR ? m - i : PG_MR;
        for(size_t ii=0;ii<mi;ii++) memset(c + (i+ii)*ldc, 0, n*sizeof(float));
        for(size_t kk=0;kk<k;kk+=PG_KC){
            size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
            pack_A_h_contig(mi, kl, a + i*lda + kk, lda, packA);
            for(size_t j=0;j<n;j+=nr){
                size_t nj=n - j < nr ? n - j : nr;
                pack_B_h_contig(kl, nj, b + kk*ldb + j, ldb, packB);
                micro(kl, packA, kl, packB, nj, c + i*ldc + j, ldc, mi, nj);
            }
        }
    }
}

void pg_cpu_hgemm(size_t m, size_t n, size_t k,
                  const uint16_t *a, size_t lda,
                  const uint16_t *b, size_t ldb,
                  float *c, size_t ldc){
    size_t nr; pg_hgemm_micro_fn micro = pg_pick_hmicro(&nr);
    size_t total=m*n*k;
    int nthreads=pg_thread_pool_size();
    if (total < (1<<18) || m < 16 || nthreads<=1) {
        for(size_t i=0;i<m;i++) memset(c + i*ldc, 0, n*sizeof(float));
        for(size_t i=0;i<m;i+=PG_MR){
            size_t mi=m - i < PG_MR ? m - i : PG_MR;
            for(size_t kk=0;kk<k;kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
                }
            }
        }
        return;
    }
    size_t nblocks=(m+PG_MR-1)/PG_MR;
    hgemm_par_t ctx={m,n,k,a,b,c,lda,ldb,ldc,micro,nr};
    pg_parallel_for(nblocks, 2, hgemm_par_fn, &ctx);
}
void pg_cpu_bgemm(size_t m, size_t n, size_t k,
                  const uint16_t *a, size_t lda,
                  const uint16_t *b, size_t ldb,
                  float *c, size_t ldc){
    size_t nr; pg_bgemm_micro_fn micro = pg_pick_bmicro(&nr);
    size_t total=m*n*k;
    int nthreads=pg_thread_pool_size();
    if (total < (1<<18) || m < 16 || nthreads<=1) {
        for(size_t i=0;i<m;i++) memset(c + i*ldc, 0, n*sizeof(float));
        for(size_t i=0;i<m;i+=PG_MR){
            size_t mi=m - i < PG_MR ? m - i : PG_MR;
            for(size_t kk=0;kk<k;kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
                }
            }
        }
        return;
    }
    size_t nblocks=(m+PG_MR-1)/PG_MR;
    bgemm_par_t ctx={m,n,k,a,b,c,lda,ldb,ldc,micro,nr};
    pg_parallel_for(nblocks, 2, bgemm_par_fn, &ctx);
}
void pg_cpu_gemm_ex(pg_dtype dtype, size_t m, size_t n, size_t k, const void *a, size_t lda, const void *b, size_t ldb, float *c, size_t ldc){
    if (dtype==PG_DTYPE_F32) pg_cpu_gemm(m,n,k,(const float*)a,lda,(const float*)b,ldb,c,ldc);
    else if (dtype==PG_DTYPE_F16) pg_cpu_hgemm(m,n,k,(const uint16_t*)a,lda,(const uint16_t*)b,ldb,c,ldc);
    else if (dtype==PG_DTYPE_BF16) pg_cpu_bgemm(m,n,k,(const uint16_t*)a,lda,(const uint16_t*)b,ldb,c,ldc);
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
    size_t nr=pg_pick_nr();
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

// fused ex for mixed: bias f32 + act after f32 accum (plan: pg_cpu_gemm_fused:402 дублировать для bias+act)
typedef struct { size_t m,n,k; const void *a,*b; const float *bias; float *c; size_t lda,ldb,ldc; int act; pg_dtype dtype; } gemm_fused_ex_t;

static void gemm_fused_ex_par(void *ctx, size_t start, size_t end){
    gemm_fused_ex_t *p=ctx;
    size_t m=p->m, n=p->n, k=p->k; int act=p->act;
    const float *bias=p->bias; float *c=p->c;
    size_t lda=p->lda,ldb=p->ldb,ldc=p->ldc;
    pg_dtype dtype=p->dtype;
    for(size_t bi=start; bi<end; ++bi){
        size_t i=bi*PG_MR; if(i>=m) break;
        size_t mi=m - i < PG_MR ? m - i : PG_MR;
        if(bias){
            for(size_t ii=0; ii<mi; ++ii){ float*crow=c + (i+ii)*ldc; for(size_t j=0;j<n;j++) crow[j]=bias[j]; }
        } else {
            for(size_t ii=0; ii<mi; ++ii) memset(c + (i+ii)*ldc, 0, n*sizeof(float));
        }
        // do gemm slice
        if (dtype==PG_DTYPE_F32){
            pg_gemm_micro_fn micro=pg_pick_micro(); size_t nr=pg_pick_nr();
            const float *a=(const float*)p->a, *b=(const float*)p->b;
            for(size_t kk=0; kk<k; kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
                }
            }
        } else if (dtype==PG_DTYPE_F16){
            size_t nr; pg_hgemm_micro_fn micro=pg_pick_hmicro(&nr);
            const uint16_t *a=(const uint16_t*)p->a, *b=(const uint16_t*)p->b;
            for(size_t kk=0; kk<k; kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
                }
            }
        } else {
            size_t nr; pg_bgemm_micro_fn micro=pg_pick_bmicro(&nr);
            const uint16_t *a=(const uint16_t*)p->a, *b=(const uint16_t*)p->b;
            for(size_t kk=0; kk<k; kk+=PG_KC){
                size_t kl=k - kk < PG_KC ? k - kk : PG_KC;
                for(size_t j=0;j<n;j+=nr){
                    size_t nj=n - j < nr ? n - j : nr;
                    micro(kl, a + i*lda + kk, lda, b + kk*ldb + j, ldb, c + i*ldc + j, ldc, mi, nj);
                }
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

void pg_cpu_gemm_fused_ex(pg_dtype dtype, size_t m, size_t n, size_t k,
                  const void *a, size_t lda,
                  const void *b, size_t ldb,
                  float *c, size_t ldc,
                  const float *bias, int act){
    size_t total=m*n*k;
    int nthreads=pg_thread_pool_size();
    if(nthreads<=1 || total < (1<<18) || m < 16){
        // serial path reuse par logic with single thread
        gemm_fused_ex_t ctx={m,n,k,a,b,bias,c,lda,ldb,ldc,act,dtype};
        gemm_fused_ex_par(&ctx, 0, (m+PG_MR-1)/PG_MR);
        return;
    }
    size_t nblocks=(m+PG_MR-1)/PG_MR;
    gemm_fused_ex_t ctx={m,n,k,a,b,bias,c,lda,ldb,ldc,act,dtype};
    pg_parallel_for(nblocks, 2, gemm_fused_ex_par, &ctx);
}
