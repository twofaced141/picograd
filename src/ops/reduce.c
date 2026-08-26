#include "reduce.h"

#include "common.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

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

    pg_tensor *out = pg_tensor_new(ondim, shape);
    if (!out)
        return NULL;

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);

    size_t ostride = keepdim && t->ndim > 1 ? out->stride[axis] : inner;

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

pg_tensor *pg_sum(const pg_tensor *t, size_t axis, bool keepdim)
{
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
    pg_tensor *out = reduce_fold(t, axis, keepdim, f_sum, 0.0f);
    if (!out)
        return NULL;
    float inv = 1.0f / (float)len;
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
    out = pg_tensor_new(ondim, shape);
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

    pg_tensor *out = pg_tensor_new(ondim, shape);
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
