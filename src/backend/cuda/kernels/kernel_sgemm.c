#define TILE 32

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

static SHARED float s_a[TILE * TILE];
static SHARED float s_b[TILE * TILE];

void pg_sgemm_kernel(const float *a, const float *b, float *c,
                     u32 m, u32 n, u32 k)
{
    const u32 ty = tid_y();
    const u32 tx = tid_x();
    const u32 row = ctaid_y() * TILE + ty;
    const u32 col = ctaid_x() * TILE + tx;

    float acc = 0.0f;
    const u32 ntiles = (k + TILE - 1) / TILE;

    for (u32 t = 0; t < ntiles; t++) {
        const u32 acol = t * TILE + tx;
        const u32 brow = t * TILE + ty;

        s_a[ty * TILE + tx] =
            (row < m && acol < k) ? a[(size_t)row * k + acol] : 0.0f;
        s_b[ty * TILE + tx] =
            (brow < k && col < n) ? b[(size_t)brow * n + col] : 0.0f;
        barrier();

        for (u32 j = 0; j < TILE; j++)
            acc += s_a[ty * TILE + j] * s_b[j * TILE + tx];
        barrier();
    }

    if (row < m && col < n)
        c[(size_t)row * n + col] = acc;
}
