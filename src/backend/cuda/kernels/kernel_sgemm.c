#define BM 64
#define BN 64
#define BK 32
#define BK4 8
#define BN4 16
#define RT 4
#define RN 4
#define TP 16
#define BK_PAD (BK + 1)
#define BN_PAD (BN + 1)

#include <stddef.h>

typedef unsigned int u32;
typedef float float4 __attribute__((ext_vector_type(4)));

static inline u32 tid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
    return r;
}

static inline u32 tid_y(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%tid.y;" : "=r"(r));
    return r;
}

static inline u32 ctaid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r));
    return r;
}

static inline u32 ctaid_y(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%ctaid.y;" : "=r"(r));
    return r;
}

static inline void barrier(void)
{
    __asm__ volatile("bar.sync 0;" ::: "memory");
}

#define SHARED __attribute__((address_space(3)))

/* double-buffered padded shared: ping-pong to halve barriers and hide latency */
static SHARED float s_a[2][BM * BK_PAD];
static SHARED float s_b[2][BK * BN_PAD];

void pg_sgemm_kernel(const float * __restrict__ a,
                     const float * __restrict__ b,
                     float * __restrict__ c,
                     u32 m, u32 n, u32 k)
{
    const u32 tx = tid_x();
    const u32 ty = tid_y();
    const u32 tid = ty * TP + tx;
    const u32 cta_y = ctaid_y();
    const u32 cta_x = ctaid_x();

    const u32 row0 = cta_y * BM + ty * RT;
    const u32 col0 = cta_x * BN + tx * RN;

    float acc[RT][RN];
#pragma unroll
    for (u32 i = 0; i < RT; i++)
#pragma unroll
        for (u32 j = 0; j < RN; j++)
            acc[i][j] = 0.0f;

    const u32 ntiles = (k + BK - 1) / BK;
    if (ntiles == 0) {
        for (u32 i = 0; i < RT; i++) {
            u32 gr = row0 + i;
            if (gr >= m) continue;
            for (u32 j = 0; j < RN; j++) {
                u32 gc = col0 + j;
                if (gc < n) c[(size_t)gr * n + gc] = 0.0f;
            }
        }
        return;
    }

    const int k_aligned = (k & 3) == 0;
    const int n_aligned = (n & 3) == 0;

    /* ---- preload tile 0 into buffer 0 ---- */
    {
        u32 gk = 0;
        int a_interior = (cta_y * BM + BM <= m) && (gk + BK <= k) && k_aligned;
        if (a_interior) {
            for (u32 v = tid; v < (u32)(BM * BK4); v += (u32)(TP * TP)) {
                u32 r = v / BK4;
                u32 vc = v % BK4;
                u32 cc = vc * 4;
                u32 gr = cta_y * BM + r;
                const float4 *src = (const float4 *)(a + (size_t)gr * k + gk + cc);
                float4 va = *src;
                u32 base = r * BK_PAD + cc;
                s_a[0][base] = va[0];
                s_a[0][base + 1] = va[1];
                s_a[0][base + 2] = va[2];
                s_a[0][base + 3] = va[3];
            }
        } else {
            for (u32 idx = tid; idx < (u32)(BM * BK); idx += (u32)(TP * TP)) {
                u32 r = idx / BK;
                u32 cc = idx % BK;
                u32 gr = cta_y * BM + r;
                u32 gc = gk + cc;
                float v = 0.0f;
                if (gr < m && gc < k) v = a[(size_t)gr * k + gc];
                s_a[0][r * BK_PAD + cc] = v;
            }
        }
        int b_interior = (gk + BK <= k) && (cta_x * BN + BN <= n) && n_aligned && k_aligned;
        if (b_interior) {
            for (u32 v = tid; v < (u32)(BK * BN4); v += (u32)(TP * TP)) {
                u32 rr = v / BN4;
                u32 vc = v % BN4;
                u32 cc = vc * 4;
                u32 gr = gk + rr;
                u32 gc = cta_x * BN + cc;
                const float4 *src = (const float4 *)(b + (size_t)gr * n + gc);
                float4 vb = *src;
                u32 base = rr * BN_PAD + cc;
                s_b[0][base] = vb[0];
                s_b[0][base + 1] = vb[1];
                s_b[0][base + 2] = vb[2];
                s_b[0][base + 3] = vb[3];
            }
        } else {
            for (u32 idx = tid; idx < (u32)(BK * BN); idx += (u32)(TP * TP)) {
                u32 rr = idx / BN;
                u32 cc = idx % BN;
                u32 gr = gk + rr;
                u32 gc = cta_x * BN + cc;
                float v = 0.0f;
                if (gr < k && gc < n) v = b[(size_t)gr * n + gc];
                s_b[0][rr * BN_PAD + cc] = v;
            }
        }
    }
    barrier();

    for (u32 t = 0; t < ntiles; t++) {
        u32 cur = t & 1;
        u32 nxt = cur ^ 1;
        u32 gk_next = (t + 1) * BK;

        /* prefetch next tile into nxt buffer while current compute */
        if (t + 1 < ntiles) {
            int a_interior = (cta_y * BM + BM <= m) && (gk_next + BK <= k) && k_aligned;
            if (a_interior) {
                for (u32 v = tid; v < (u32)(BM * BK4); v += (u32)(TP * TP)) {
                    u32 r = v / BK4;
                    u32 vc = v % BK4;
                    u32 cc = vc * 4;
                    u32 gr = cta_y * BM + r;
                    const float4 *src = (const float4 *)(a + (size_t)gr * k + gk_next + cc);
                    float4 va = *src;
                    u32 base = r * BK_PAD + cc;
                    s_a[nxt][base] = va[0];
                    s_a[nxt][base + 1] = va[1];
                    s_a[nxt][base + 2] = va[2];
                    s_a[nxt][base + 3] = va[3];
                }
            } else {
                for (u32 idx = tid; idx < (u32)(BM * BK); idx += (u32)(TP * TP)) {
                    u32 r = idx / BK;
                    u32 cc = idx % BK;
                    u32 gr = cta_y * BM + r;
                    u32 gc = gk_next + cc;
                    float v = 0.0f;
                    if (gr < m && gc < k) v = a[(size_t)gr * k + gc];
                    s_a[nxt][r * BK_PAD + cc] = v;
                }
            }
            int b_interior = (gk_next + BK <= k) && (cta_x * BN + BN <= n) && n_aligned && k_aligned;
            if (b_interior) {
                for (u32 v = tid; v < (u32)(BK * BN4); v += (u32)(TP * TP)) {
                    u32 rr = v / BN4;
                    u32 vc = v % BN4;
                    u32 cc = vc * 4;
                    u32 gr = gk_next + rr;
                    u32 gc = cta_x * BN + cc;
                    const float4 *src = (const float4 *)(b + (size_t)gr * n + gc);
                    float4 vb = *src;
                    u32 base = rr * BN_PAD + cc;
                    s_b[nxt][base] = vb[0];
                    s_b[nxt][base + 1] = vb[1];
                    s_b[nxt][base + 2] = vb[2];
                    s_b[nxt][base + 3] = vb[3];
                }
            } else {
                for (u32 idx = tid; idx < (u32)(BK * BN); idx += (u32)(TP * TP)) {
                    u32 rr = idx / BN;
                    u32 cc = idx % BN;
                    u32 gr = gk_next + rr;
                    u32 gc = cta_x * BN + cc;
                    float v = 0.0f;
                    if (gr < k && gc < n) v = b[(size_t)gr * n + gc];
                    s_b[nxt][rr * BN_PAD + cc] = v;
                }
            }
        }

        /* compute current tile */
#pragma unroll 1
        for (u32 kk = 0; kk < BK; kk++) {
            float ra[RT], rb[RN];
#pragma unroll
            for (u32 i = 0; i < RT; i++)
                ra[i] = s_a[cur][(ty * RT + i) * BK_PAD + kk];
#pragma unroll
            for (u32 j = 0; j < RN; j++)
                rb[j] = s_b[cur][kk * BN_PAD + tx * RN + j];
#pragma unroll
            for (u32 i = 0; i < RT; i++)
#pragma unroll
                for (u32 j = 0; j < RN; j++)
                    acc[i][j] += ra[i] * rb[j];
        }
        /* ensure next tile is fully loaded and current compute done */
        if (t + 1 < ntiles)
            barrier();
    }

    for (u32 i = 0; i < RT; i++) {
        const u32 gr = row0 + i;
        if (gr >= m) continue;
        for (u32 j = 0; j < RN; j++) {
            const u32 gc = col0 + j;
            if (gc < n) c[(size_t)gr * n + gc] = acc[i][j];
        }
    }
}
