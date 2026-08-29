#include "reduce.h"

#include "common.h"
#include "../backend/backend.h"
#include "../thread/pool.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const float *src; float *dst; size_t outer; size_t len; size_t inner; size_t ostride; } reduce_sum_par_t;
static void reduce_sum_par_fn(void *ctx, size_t s, size_t e){
    reduce_sum_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        float *dptr = p->dst + o * p->ostride;
        const float *base = p->src + o * p->len * p->inner;
        for(size_t ii=0;ii<p->inner;ii++) dptr[ii]=0.0f;
        for(size_t v=0;v<p->len;v++){
            const float *sptr = base + v * p->inner;
            #pragma GCC ivdep
            for(size_t ii=0;ii<p->inner;ii++) dptr[ii]+=sptr[ii];
        }
    }
}
typedef struct { const float *src; float *dst; size_t outer; size_t len; size_t inner; size_t ostride; } reduce_max_par_t;
static void reduce_max_par_fn(void *ctx, size_t s, size_t e){
    reduce_max_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        float *dptr = p->dst + o * p->ostride;
        const float *base = p->src + o * p->len * p->inner;
        for(size_t ii=0;ii<p->inner;ii++) dptr[ii]=base[ii];
        for(size_t v=1;v<p->len;v++){
            const float *sptr=base+v*p->inner;
            #pragma GCC ivdep
            for(size_t ii=0;ii<p->inner;ii++) if(sptr[ii]>dptr[ii]) dptr[ii]=sptr[ii];
        }
    }
}
static void reduce_min_par_fn(void *ctx, size_t s, size_t e){
    reduce_max_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        float *dptr = p->dst + o * p->ostride;
        const float *base = p->src + o * p->len * p->inner;
        for(size_t ii=0;ii<p->inner;ii++) dptr[ii]=base[ii];
        for(size_t v=1;v<p->len;v++){
            const float *sptr=base+v*p->inner;
            #pragma GCC ivdep
            for(size_t ii=0;ii<p->inner;ii++) if(sptr[ii]<dptr[ii]) dptr[ii]=sptr[ii];
        }
    }
}
typedef struct { const float *src; float *dst; size_t outer; size_t len; } reduce_sum1_par_t;
static void reduce_sum1_par_fn(void *ctx, size_t s, size_t e){
    reduce_sum1_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        const float *src=p->src+o*p->len;
        float *dst=p->dst+o;
        float acc=0.0f;
        #pragma GCC ivdep
        for(size_t v=0;v<p->len;v++) acc+=src[v];
        *dst=acc;
    }
}
typedef struct { const float *src; float *dst; size_t outer; size_t len; float init; float (*f)(float,float); } reduce_fold1_par_t;
static void reduce_fold1_par_fn(void *ctx, size_t s, size_t e){
    reduce_fold1_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        const float *src=p->src+o*p->len;
        float acc=p->init;
        for(size_t v=0;v<p->len;v++) acc=p->f(acc,src[v]);
        p->dst[o]=acc;
    }
}

static void out_shape_for(const pg_tensor *t, size_t axis, bool keepdim, size_t *shape)
{
    if (keepdim) {
        for (size_t d = 0; d < t->ndim; d++)
            shape[d] = d == axis ? 1 : t->shape[d];
        return;
    }
    size_t j = 0;
    for (size_t d = 0; d < t->ndim; d++)
        if (d != axis)
            shape[j++] = t->shape[d];
    if (j == 0)
        shape[j++] = 1;
}

typedef float (*combine_fn)(float acc, float x);

static float f_sum(float a, float x) { return a + x; }
static float f_max(float a, float x) { return x > a ? x : a; }
static float f_min(float a, float x) { return x < a ? x : a; }

static pg_tensor *reduce_fold(const pg_tensor *t, size_t axis, bool keepdim,
                              combine_fn f, float init)
{
    assert(t && t->data && t->ndim >= 1 && axis < t->ndim);

    size_t shape[PG_MAX_NDIM];
    out_shape_for(t, axis, keepdim, shape);
    size_t ondim = keepdim || t->ndim > 1 ? (keepdim ? t->ndim : t->ndim - 1) : 1;

    pg_tensor *out = pg_tensor_empty(ondim, shape);
    if (!out)
        return NULL;

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);

    size_t ostride = keepdim && t->ndim > 1 ? out->stride[axis] : inner;

    // Fast path: inner == 1 => contiguous reduction along last axis (most common)
    if (inner == 1) {
        if (outer >= 32 && outer * len >= 32768) {
            // parallel over outer
            if (f == f_sum) {
                reduce_sum1_par_t ctx={t->data, out->data, outer, len};
                pg_parallel_for(outer, 32, reduce_sum1_par_fn, &ctx);
            } else {
                reduce_fold1_par_t ctx={t->data, out->data, outer, len, init, f};
                pg_parallel_for(outer, 32, reduce_fold1_par_fn, &ctx);
            }
            return out;
        }
        for (size_t o = 0; o < outer; o++) {
            const float *src = t->data + o * len;
            float *dst = out->data + o * (ostride ? ostride : 1);
            float acc = init;
            for (size_t v = 0; v < len; v++) acc = f(acc, src[v]);
            *dst = acc;
        }
        if (outer > 0 && inner == 1) {
            return out;
        }
    }

    // Cache-friendly blocked version for inner > 1 (e.g., axis=0 column sum)
    if (inner > 1) {
        // initialize out
        for (size_t i = 0; i < out->numel; i++) out->data[i] = init;
        if (f == f_sum) {
            if (outer >= 8 && outer * inner * len >= 16384) {
                reduce_sum_par_t ctx={t->data, out->data, outer, len, inner, ostride};
                pg_parallel_for(outer, 8, reduce_sum_par_fn, &ctx);
            } else {
                for (size_t o = 0; o < outer; o++) {
                    float *dptr = out->data + o * ostride;
                    const float *base = t->data + o * len * inner;
                    for (size_t v = 0; v < len; v++) {
                        const float *sptr = base + v * inner;
                        #pragma GCC ivdep
                        for (size_t ii = 0; ii < inner; ii++) dptr[ii] += sptr[ii];
                    }
                }
            }
            return out;
        } else if (f == f_max) {
            if (outer >= 8 && outer * inner * len >= 16384) {
                reduce_max_par_t ctx={t->data, out->data, outer, len, inner, ostride};
                pg_parallel_for(outer, 8, reduce_max_par_fn, &ctx);
            } else {
                for (size_t o = 0; o < outer; o++) {
                    float *dptr = out->data + o * ostride;
                    const float *base = t->data + o * len * inner;
                    for (size_t ii = 0; ii < inner; ii++) dptr[ii] = base[ii];
                    for (size_t v = 1; v < len; v++) {
                        const float *sptr = base + v * inner;
                        #pragma GCC ivdep
                        for (size_t ii = 0; ii < inner; ii++) if (sptr[ii] > dptr[ii]) dptr[ii] = sptr[ii];
                    }
                }
            }
            return out;
        } else if (f == f_min) {
            if (outer >= 8 && outer * inner * len >= 16384) {
                reduce_max_par_t ctx={t->data, out->data, outer, len, inner, ostride};
                pg_parallel_for(outer, 8, reduce_min_par_fn, &ctx);
            } else {
                for (size_t o = 0; o < outer; o++) {
                    float *dptr = out->data + o * ostride;
                    const float *base = t->data + o * len * inner;
                    for (size_t ii = 0; ii < inner; ii++) dptr[ii] = base[ii];
                    for (size_t v = 1; v < len; v++) {
                        const float *sptr = base + v * inner;
                        #pragma GCC ivdep
                        for (size_t ii = 0; ii < inner; ii++) if (sptr[ii] < dptr[ii]) dptr[ii] = sptr[ii];
                    }
                }
            }
            return out;
        }
        // unsupported combine function -> generic path below
    }

    for (size_t o = 0; o < outer; o++) {
        for (size_t ii = 0; ii < inner; ii++) {
            float acc = init;
            const float *p = t->data + o * len * inner + ii;
            for (size_t v = 0; v < len; v++, p += inner)
                acc = f(acc, *p);
            out->data[o * ostride + ii] = acc;
        }
    }
    return out;
}

static pg_tensor *try_sum_gpu(const pg_tensor *t, size_t axis, bool keepdim, float scale)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;
    if (t->numel > UINT_MAX)
        return NULL;
    size_t shape[PG_MAX_NDIM];
    out_shape_for(t, axis, keepdim, shape);
    size_t ondim = keepdim || t->ndim > 1 ? (keepdim ? t->ndim : t->ndim - 1) : 1;
    if (ondim > PG_MAX_NDIM)
        return NULL;

    pg_tensor *out = pg_tensor_empty(ondim, shape);
    if (!out)
        return NULL;
    if (out->numel > UINT_MAX)
        return NULL;

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);
    if (outer > UINT_MAX || len > UINT_MAX || inner > UINT_MAX)
    {
        pg_tensor_free(out);
        return NULL;
    }
    size_t ostride = keepdim && t->ndim > 1 ? out->stride[axis] : inner;
    if (ostride > UINT_MAX)
    {
        pg_tensor_free(out);
        return NULL;
    }

    size_t bytes_src = t->numel * sizeof(float);
    size_t bytes_out = out->numel * sizeof(float);

    float *da = pg_dev_malloc(bytes_src ? bytes_src : 1);
    float *dc = pg_dev_malloc(bytes_out ? bytes_out : 1);
    if (!da || !dc) {
        if (da) pg_dev_free(da);
        if (dc) pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da, t->data, bytes_src) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    /* kernel writes out = sum*scale directly; no need to pre-zero */

    pg_status st = pg_op_sum_axis(dc, da, scale, outer, len, inner, ostride);
    if (st != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_dev_sync() != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data, dc, bytes_out) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_free(da);
    pg_dev_free(dc);
    return out;
}

pg_tensor *pg_sum(const pg_tensor *t, size_t axis, bool keepdim)
{
    pg_tensor *g = try_sum_gpu(t, axis, keepdim, 1.0f);
    if (g) return g;
    return reduce_fold(t, axis, keepdim, f_sum, 0.0f);
}

pg_tensor *pg_max(const pg_tensor *t, size_t axis, bool keepdim)
{
    return reduce_fold(t, axis, keepdim, f_max, -INFINITY);
}

pg_tensor *pg_min(const pg_tensor *t, size_t axis, bool keepdim)
{
    return reduce_fold(t, axis, keepdim, f_min, INFINITY);
}

pg_tensor *pg_mean(const pg_tensor *t, size_t axis, bool keepdim)
{
    assert(t && t->data && t->ndim >= 1 && axis < t->ndim);
    size_t len = t->shape[axis];
    float inv = 1.0f / (float)len;
    pg_tensor *g = try_sum_gpu(t, axis, keepdim, inv);
    if (g) return g;
    pg_tensor *out = reduce_fold(t, axis, keepdim, f_sum, 0.0f);
    if (!out)
        return NULL;
    for (size_t i = 0; i < out->numel; i++)
        out->data[i] *= inv;
    return out;
}

static pg_tensor *reduce_moment(const pg_tensor *t, size_t axis, bool keepdim,
                                int ddof, float (*post)(float))
{
    assert(ddof == 0 || ddof == 1);
    size_t len = t->shape[axis];
    if ((size_t)ddof >= len)
        return NULL;

    pg_tensor *mean = pg_mean(t, axis, true);
    if (!mean)
        return NULL;

    size_t outer, alen, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &alen, &inner);

    size_t mstride = t->ndim > 1 ? mean->stride[axis] : inner;

    pg_tensor *out;
    size_t shape[PG_MAX_NDIM];
    out_shape_for(t, axis, keepdim, shape);
    size_t ondim = keepdim ? t->ndim : (t->ndim > 1 ? t->ndim - 1 : 1);
    out = pg_tensor_empty(ondim, shape);
    if (!out) {
        pg_tensor_free(mean);
        return NULL;
    }

    float inv = 1.0f / (float)(len - (size_t)ddof);
    for (size_t o = 0; o < outer; o++) {
        for (size_t ii = 0; ii < inner; ii++) {
            float m = mean->data[o * mstride + ii];
            float acc = 0.0f;
            const float *p = t->data + o * alen * inner + ii;
            for (size_t v = 0; v < alen; v++, p += inner) {
                float d = *p - m;
                acc += d * d;
            }
            out->data[o * (keepdim && t->ndim > 1 ? out->stride[axis] : inner) + ii] =
                post ? post(acc * inv) : acc * inv;
        }
    }
    pg_tensor_free(mean);
    return out;
}

static float fsqrt(float x) { return sqrtf(x); }

pg_tensor *pg_var(const pg_tensor *t, size_t axis, bool keepdim, int ddof)
{
    return reduce_moment(t, axis, keepdim, ddof, NULL);
}

pg_tensor *pg_std(const pg_tensor *t, size_t axis, bool keepdim, int ddof)
{
    return reduce_moment(t, axis, keepdim, ddof, fsqrt);
}

static pg_tensor *reduce_arg(const pg_tensor *t, size_t axis, bool keepdim, bool largest)
{
    assert(t && t->data && t->ndim >= 1 && axis < t->ndim);

    size_t shape[PG_MAX_NDIM];
    out_shape_for(t, axis, keepdim, shape);
    size_t ondim = keepdim ? t->ndim : (t->ndim > 1 ? t->ndim - 1 : 1);

    pg_tensor *out = pg_tensor_empty(ondim, shape);
    if (!out)
        return NULL;

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);
    size_t ostride = keepdim && t->ndim > 1 ? out->stride[axis] : inner;

    for (size_t o = 0; o < outer; o++) {
        for (size_t ii = 0; ii < inner; ii++) {
            const float *p = t->data + o * len * inner + ii;
            float best = *p;
            size_t besti = 0;
            for (size_t v = 1; v < len; v++) {
                float x = p[v * inner];
                if (largest ? x > best : x < best) {
                    best = x;
                    besti = v;
                }
            }
            out->data[o * ostride + ii] = (float)besti;
        }
    }
    return out;
}

pg_tensor *pg_argmax(const pg_tensor *t, size_t axis, bool keepdim)
{
    return reduce_arg(t, axis, keepdim, true);
}

pg_tensor *pg_argmin(const pg_tensor *t, size_t axis, bool keepdim)
{
    return reduce_arg(t, axis, keepdim, false);
}
