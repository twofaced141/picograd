#include "activations.h"

#include "common.h"
#include <assert.h>
#include <math.h>

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

static float frelu(float x) { return x > 0.0f ? x : 0.0f; }
static float fsigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float fgelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); }

pg_tensor *pg_relu(const pg_tensor *a)
{
    return map_inplace(a, frelu);
}

pg_tensor *pg_sigmoid(const pg_tensor *a)
{
    return map_inplace(a, fsigmoid);
}

pg_tensor *pg_tanh(const pg_tensor *a)
{
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
    return softmax_impl(t, axis, false);
}

pg_tensor *pg_log_softmax(const pg_tensor *t, size_t axis)
{
    return softmax_impl(t, axis, true);
}
