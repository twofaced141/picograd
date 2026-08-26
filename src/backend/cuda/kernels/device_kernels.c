#include "../../backend.h"

typedef unsigned int u32;

static inline u32 tid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
    return r;
}

static inline u32 ctaid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r));
    return r;
}

static inline u32 ntid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%ntid.x;" : "=r"(r));
    return r;
}

static inline u32 nctaid_x(void)
{
    u32 r;
    __asm__ volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r));
    return r;
}

static float pg_expf(float x)
{
    if (x > 88.0f)
        return 3.4028235e38f;
    if (x < -88.0f)
        return 0.0f;

    const float LOG2E = 1.442695041f;
    const float LN2 = 0.693147181f;

    float y = x * LOG2E;
    int k = (int)(y >= 0.0f ? y + 0.5f : y - 0.5f);
    float r = x - (float)k * LN2;

    float p = 1.0f
              + r * (1.0f
                     + r * (0.5f
                            + r * (0.166666667f
                                   + r * (0.041666667f
                                          + r * (0.008333333f
                                                 + r * 0.001388889f)))));

    union {
        unsigned u;
        float f;
    } sc;
    sc.u = (unsigned)((k + 127) << 23);
    return p * sc.f;
}

static float pg_logf(float x)
{
    union {
        unsigned u;
        float f;
    } v;
    v.f = x;
    int e = (int)(v.u >> 23) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u;
    float m = v.f;

    if (m > 1.414213562f) {
        m *= 0.5f;
        e++;
    }

    float s = (m - 1.0f) / (m + 1.0f);
    float z = s * s;
    float p = s
              * (2.0f
                 + z * (0.666666667f
                        + z * (0.4f
                               + z * (0.285714286f
                                      + z * (0.222222222f
                                             + z * 0.181818182f)))));
    return p + (float)e * 0.693147181f;
}

static float pg_sinf(float x)
{
    const float TWO_OVER_PI = 0.636619772f;
    const float PI_2A = 1.5707855225f;
    const float PI_2B = 7.1386587e-7f;

    float q = x * TWO_OVER_PI;
    int quad = (int)(q >= 0.0f ? q + 0.5f : q - 0.5f);
    float r = x - (float)quad * PI_2A - (float)quad * PI_2B;
    float r2 = r * r;

    float sp = r
               + r * r2
                     * (-0.166666567f
                        + r2 * (0.008333025f - r2 * 0.000198074f));
    float cp = 1.0f
               + r2 * (-0.499999925f
                       + r2 * (0.041645982f - r2 * 0.001358558f));

    switch (quad & 3) {
    case 0:
        return sp;
    case 1:
        return cp;
    case 2:
        return -sp;
    default:
        return -cp;
    }
}

static float pg_cosf(float x)
{
    return pg_sinf(x + 1.570796327f);
}

static float pg_tanhf(float x)
{
    if (x > 10.0f)
        return 1.0f;
    if (x < -10.0f)
        return -1.0f;
    float e2 = pg_expf(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

static float pg_erff(float x)
{
    float ax = x < 0.0f ? -x : x;
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = t * (0.254829592f
                   + t * (-0.284496736f
                          + t * (1.421413741f
                                 + t * (-1.453152027f + t * 1.061405429f))));
    float r = 1.0f - p * pg_expf(-x * x);
    return x < 0.0f ? -r : r;
}

static float pg_sqrtf(float x)
{
    if (x <= 0.0f)
        return 0.0f;

    union {
        unsigned u;
        float f;
    } v;
    v.f = x;
    v.u = (v.u >> 1) + 0x1fc00000u;

    float g = v.f;
    for (int it = 0; it < 4; it++)
        g = 0.5f * (g + x / g);
    return g;
}

static float map_apply(int op, float x)
{
    switch (op) {
    case PG_MAP_EXP:
        return pg_expf(x);
    case PG_MAP_LOG:
        return pg_logf(x);
    case PG_MAP_SIN:
        return pg_sinf(x);
    case PG_MAP_COS:
        return pg_cosf(x);
    case PG_MAP_SQRT:
        return pg_sqrtf(x);
    case PG_MAP_NEG:
        return -x;
    case PG_MAP_ABS:
        return x < 0.0f ? -x : x;
    case PG_MAP_ERF:
        return pg_erff(x);
    case PG_MAP_RELU:
        return x > 0.0f ? x : 0.0f;
    case PG_MAP_SIGMOID:
        return 1.0f / (1.0f + pg_expf(-x));
    case PG_MAP_TANH:
        return pg_tanhf(x);
    default:
        return x;
    }
}

static float bin_apply(int op, float a, float b)
{
    switch (op) {
    case PG_BIN_ADD:
        return a + b;
    case PG_BIN_SUB:
        return a - b;
    case PG_BIN_MUL:
        return a * b;
    case PG_BIN_DIV:
        return a / b;
    case PG_BIN_SIG_BW:
        return b * a * (1.0f - a);
    case PG_BIN_TANH_BW:
        return b * (1.0f - a * a);
    case PG_BIN_RELU_BW:
        return a > 0.0f ? b : 0.0f;
    default:
        return a;
    }
}

void pg_k_map(float *out, const float *src, u32 n, u32 op)
{
    const u32 stride = ntid_x() * nctaid_x();
    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < n; i += stride)
        out[i] = map_apply((int)op, src[i]);
}

void pg_k_bin(float *out, const float *a, const float *b, u32 op,
              pg_k_bin_args ar)
{
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < ar.numel;
         i += stride) {
        u32 idx[PG_MAX_OP_NDIM];
        u32 rem = i;
        for (u32 d = ar.ndim; d-- > 0;) {
            idx[d] = rem % ar.shape[d];
            rem /= ar.shape[d];
        }

        u32 oa = 0, ob = 0;
        for (u32 d = 0; d < ar.ndim; d++) {
            oa += idx[d] * ar.sa[d];
            ob += idx[d] * ar.sb[d];
        }

        out[i] = bin_apply((int)op, a[oa], b[ob]);
    }
}

/* dst (larger domain) += scale * src gathered via s[] strides */
/* ar.shape/numel describe DST; ar.s are SRC strides per dst dim */
void pg_k_accum_gather(float *dst, const float *src, float scale,
                       pg_k_strides ar)
{
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < ar.numel;
         i += stride) {
        u32 idx[PG_MAX_OP_NDIM];
        u32 rem = i;
        for (u32 d = ar.ndim; d-- > 0;) {
            idx[d] = rem % ar.shape[d];
            rem /= ar.shape[d];
        }

        u32 os = 0;
        for (u32 d = 0; d < ar.ndim; d++)
            os += idx[d] * ar.s[d];

        dst[i] += scale * src[os];
    }
}

/* dst (projected via s[]) += scale * src iterating src's own domain */
/* ar.shape/numel describe SRC; ar.s are DST strides per src dim */
void pg_k_accum_scatter(float *dst, const float *src, float scale,
                        pg_k_strides ar)
{
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < ar.numel;
         i += stride) {
        u32 idx[PG_MAX_OP_NDIM];
        u32 rem = i;
        for (u32 d = ar.ndim; d-- > 0;) {
            idx[d] = rem % ar.shape[d];
            rem /= ar.shape[d];
        }

        u32 od = 0;
        for (u32 d = 0; d < ar.ndim; d++)
            od += idx[d] * ar.s[d];

        dst[od] += scale * src[i];
    }
}

void pg_k_sum_axis(float *out, const float *src, float scale,
                   u32 outer, u32 len, u32 inner, u32 ostride)
{
    const u32 total = outer * inner;
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < total; i += stride) {
        const u32 o = i / inner;
        const u32 ii = i % inner;
        float acc = 0.0f;
        const float *p = src + (size_t)o * len * inner + ii;
        for (u32 j = 0; j < len; j++, p += inner)
            acc += *p;
        out[(size_t)o * ostride + ii] = acc * scale;
    }
}

void pg_k_softmax(float *out, const float *src, u32 outer, u32 len,
                  u32 inner)
{
    const u32 total = outer * inner;
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < total; i += stride) {
        const u32 o = i / inner;
        const u32 ii = i % inner;
        const float *col = src + (size_t)o * len * inner + ii;
        float *dcol = out + (size_t)o * len * inner + ii;

        float mx = col[0];
        for (u32 j = 1; j < len; j++) {
            float v = col[(size_t)j * inner];
            if (v > mx)
                mx = v;
        }
        float sum = 0.0f;
        for (u32 j = 0; j < len; j++) {
            float e = pg_expf(col[(size_t)j * inner] - mx);
            dcol[(size_t)j * inner] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (u32 j = 0; j < len; j++)
            dcol[(size_t)j * inner] *= inv;
    }
}

void pg_k_copy_strided(float *dst, const float *src, pg_k_strides ar)
{
    const u32 stride = ntid_x() * nctaid_x();

    for (u32 i = tid_x() + ctaid_x() * ntid_x(); i < ar.numel;
         i += stride) {
        u32 idx[PG_MAX_OP_NDIM];
        u32 rem = i;
        for (u32 d = ar.ndim; d-- > 0;) {
            idx[d] = rem % ar.shape[d];
            rem /= ar.shape[d];
        }

        u32 os = 0;
        for (u32 d = 0; d < ar.ndim; d++)
            os += idx[d] * ar.s[d];

        dst[i] = src[os];
    }
}
