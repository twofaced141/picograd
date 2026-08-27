#define BM 64
#define BN 64
#define BK 32
#define RT 4
#define RN 4
#define TP 16

#include <stddef.h>

typedef unsigned int u32;

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

static SHARED float s_a[BM * BK];
static SHARED float s_b[BK * BN];

void pg_sgemm_kernel(const float *a, const float *b, float *c,
                     u32 m, u32 n, u32 k)
{
    const u32 tx = tid_x();
    const u32 ty = tid_y();
    const u32 tid = ty * TP + tx;

    const u32 row0 = ctaid_y() * BM + ty * RT;
    const u32 col0 = ctaid_x() * BN + tx * RN;

    float acc[RT][RN];
    for (u32 i = 0; i < RT; i++)
        for (u32 j = 0; j < RN; j++)
            acc[i][j] = 0.0f;

    const u32 ntiles = (k + BK - 1) / BK;

    for (u32 t = 0; t < ntiles; t++) {
        const u32 gk = t * BK;

        for (u32 idx = tid; idx < BM * BK; idx += TP * TP) {
            const u32 r = idx / BK;
            const u32 cc = idx % BK;
            const u32 gr = ctaid_y() * BM + r;
            const u32 gc = gk + cc;
            s_a[idx] =
                (gr < m && gc < k) ? a[(size_t)gr * k + gc] : 0.0f;
        }
        for (u32 idx = tid; idx < BK * BN; idx += TP * TP) {
            const u32 rr = idx / BN;
            const u32 cc = idx % BN;
            const u32 gr = gk + rr;
            const u32 gc = ctaid_x() * BN + cc;
            s_b[idx] =
                (gr < k && gc < n) ? b[(size_t)gr * n + gc] : 0.0f;
        }
        barrier();

        for (u32 kk = 0; kk < BK; kk++) {
            float ra[RT], rb[RN];
            for (u32 i = 0; i < RT; i++)
                ra[i] = s_a[(ty * RT + i) * BK + kk];
            for (u32 j = 0; j < RN; j++)
                rb[j] = s_b[kk * BN + tx * RN + j];
            for (u32 i = 0; i < RT; i++)
                for (u32 j = 0; j < RN; j++)
                    acc[i][j] += ra[i] * rb[j];
        }
        barrier();
    }

    for (u32 i = 0; i < RT; i++) {
        const u32 gr = row0 + i;
        if (gr >= m)
            continue;
        for (u32 j = 0; j < RN; j++) {
            const u32 gc = col0 + j;
            if (gc < n)
                c[(size_t)gr * n + gc] = acc[i][j];
        }
    }
}
