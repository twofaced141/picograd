#include "activations.h"

#include "common.h"
#include "../backend/backend.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static pg_tensor *map_inplace(const pg_tensor *a, float (*f)(float))
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    for (size_t i = 0; i < r->numel; i++)
        r->data[i] = f(r->data[i]);
    return r;
}

static pg_tensor *try_map_gpu_act(const pg_tensor *a, int op)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;
    if (a->numel > UINT_MAX)
        return NULL;
    pg_tensor *out = pg_tensor_new(a->ndim, a->shape);
    if (!out)
        return NULL;
    size_t bytes = a->numel * sizeof(float);
    float *da = pg_dev_malloc(bytes ? bytes : 1);
    float *dc = pg_dev_malloc(bytes ? bytes : 1);
    if (!da || !dc) {
        if (da) pg_dev_free(da);
        if (dc) pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da, a->data, bytes) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_status st = pg_op_map(dc, da, a->numel, op);
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
    if (pg_copy_d2h(out->data, dc, bytes) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_free(da);
    pg_dev_free(dc);
    return out;
}

static float frelu(float x) { return x > 0.0f ? x : 0.0f; }
static float fsigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float fgelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); }

pg_tensor *pg_relu(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_RELU);
    if (g) return g;
    return map_inplace(a, frelu);
}

pg_tensor *pg_sigmoid(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_SIGMOID);
    if (g) return g;
    return map_inplace(a, fsigmoid);
}

pg_tensor *pg_tanh(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_TANH);
    if (g) return g;
    return map_inplace(a, tanhf);
}

pg_tensor *pg_gelu(const pg_tensor *a)
{
    return map_inplace(a, fgelu);
}

pg_tensor *pg_leaky_relu(const pg_tensor *a, float alpha)
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    for (size_t i = 0; i < r->numel; i++) {
        float v = r->data[i];
        r->data[i] = v > 0.0f ? v : alpha * v;
    }
    return r;
}

static pg_tensor *softmax_impl(const pg_tensor *t, size_t axis, bool log)
{
    assert(t && t->data && axis < t->ndim);

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);

    pg_tensor *r = pg_tensor_clone(t);
    if (!r)
        return NULL;

    const float *src = t->data;
    float *dst = r->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            const float *scol = src + o * len * inner + i;
            float *dcol = dst + o * len * inner + i;
            float m = -INFINITY;
            for (size_t j = 0; j < len; j++) {
                float v = scol[j * inner];
                if (v > m)
                    m = v;
            }
            float s = 0.0f;
            for (size_t j = 0; j < len; j++)
                s += expf(scol[j * inner] - m);
            if (log) {
                float ls = logf(s);
                for (size_t j = 0; j < len; j++)
                    dcol[j * inner] = scol[j * inner] - m - ls;
            } else {
                float inv = 1.0f / s;
                for (size_t j = 0; j < len; j++)
                    dcol[j * inner] = expf(scol[j * inner] - m) * inv;
            }
        }
    }
    return r;
}

pg_tensor *pg_softmax(const pg_tensor *t, size_t axis)
{
    if (pg_get_device() != PG_DEV_CPU && t->numel <= UINT_MAX) {
        size_t outer, len, inner;
        pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);
        if (outer <= UINT_MAX && len <= UINT_MAX && inner <= UINT_MAX) {
            pg_tensor *out = pg_tensor_new(t->ndim, t->shape);
            if (out) {
                size_t bytes = t->numel * sizeof(float);
                float *da = pg_dev_malloc(bytes ? bytes : 1);
                float *dc = pg_dev_malloc(bytes ? bytes : 1);
                if (da && dc &&
                    pg_copy_h2d(da, t->data, bytes) == PG_OK &&
                    pg_op_softmax(dc, da, outer, len, inner) == PG_OK &&
                    pg_dev_sync() == PG_OK &&
                    pg_copy_d2h(out->data, dc, bytes) == PG_OK) {
                    pg_dev_free(da);
                    pg_dev_free(dc);
                    return out;
                }
                if (da) pg_dev_free(da);
                if (dc) pg_dev_free(dc);
                pg_tensor_free(out);
            }
        }
    }
    return softmax_impl(t, axis, false);
}

pg_tensor *pg_log_softmax(const pg_tensor *t, size_t axis)
{
    return softmax_impl(t, axis, true);
}
