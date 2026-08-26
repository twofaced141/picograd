#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool valid_shape(size_t ndim, const size_t *shape)
{
    if (ndim == 0 || ndim > PG_MAX_NDIM || shape == NULL)
        return false;
    for (size_t i = 0; i < ndim; i++)
        if (shape[i] == 0)
            return false;
    return true;
}

static void compute_strides(size_t ndim, const size_t *shape, size_t *stride)
{
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0;) {
        stride[i] = acc;
        acc *= shape[i];
    }
}

pg_tensor *pg_tensor_new(size_t ndim, const size_t *shape)
{
    if (!valid_shape(ndim, shape))
        return NULL;

    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (numel > SIZE_MAX / shape[i])
            return NULL;
        numel *= shape[i];
    }
    if (numel > SIZE_MAX / sizeof(float))
        return NULL;

    pg_tensor *t = malloc(sizeof(*t));
    if (!t)
        return NULL;

    t->data = calloc(numel, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }

    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, t->shape, t->stride);
    t->numel = numel;
    return t;
}

void pg_tensor_free(pg_tensor *t)
{
    if (!t)
        return;
    free(t->data);
    free(t);
}

pg_tensor *pg_tensor_full(size_t ndim, const size_t *shape, float value)
{
    pg_tensor *t = pg_tensor_new(ndim, shape);
    if (!t)
        return NULL;
    pg_tensor_fill(t, value);
    return t;
}

pg_tensor *pg_tensor_zeros(size_t ndim, const size_t *shape)
{
    return pg_tensor_new(ndim, shape);
}

pg_tensor *pg_tensor_ones(size_t ndim, const size_t *shape)
{
    return pg_tensor_full(ndim, shape, 1.0f);
}

pg_tensor *pg_tensor_from_data(size_t ndim, const size_t *shape, const float *data)
{
    if (!data)
        return NULL;
    pg_tensor *t = pg_tensor_new(ndim, shape);
    if (!t)
        return NULL;
    memcpy(t->data, data, t->numel * sizeof(float));
    return t;
}

pg_tensor *pg_tensor_arange(float start, float stop, float step)
{
    if (step <= 0.0f || stop <= start)
        return NULL;
    size_t n = (size_t)ceilf((stop - start) / step);
    pg_tensor *t = pg_tensor_new(1, &n);
    if (!t)
        return NULL;
    for (size_t i = 0; i < n; i++)
        t->data[i] = start + step * (float)i;
    return t;
}

static unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;

void pg_seed(unsigned long long seed)
{
    rng_state = seed != 0 ? seed : 0x9E3779B97F4A7C15ULL;
}

static unsigned long long rng_next(void)
{
    rng_state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rng_u01(void)
{
    return (float)(rng_next() >> 40) * (1.0f / 16777216.0f);
}

pg_tensor *pg_tensor_uniform(size_t ndim, const size_t *shape, float low, float high)
{
    pg_tensor *t = pg_tensor_new(ndim, shape);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] = low + rng_u01() * (high - low);
    return t;
}

pg_tensor *pg_tensor_normal(size_t ndim, const size_t *shape, float mean, float stddev)
{
    pg_tensor *t = pg_tensor_new(ndim, shape);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++) {
        float u1 = rng_u01();
        float u2 = rng_u01();
        if (u1 < 1e-12f)
            u1 = 1e-12f;
        float mag = stddev * sqrtf(-2.0f * logf(u1));
        t->data[i] = mean + mag * cosf(2.0f * 3.14159265358979f * u2);
    }
    return t;
}

pg_tensor *pg_tensor_linspace(float start, float stop, size_t num)
{
    if (num == 0)
        return NULL;
    pg_tensor *t = pg_tensor_new(1, &num);
    if (!t)
        return NULL;
    if (num == 1) {
        t->data[0] = start;
        return t;
    }
    float step = (stop - start) / (float)(num - 1);
    for (size_t i = 0; i < num - 1; i++)
        t->data[i] = start + step * (float)i;
    t->data[num - 1] = stop;
    return t;
}

pg_tensor *pg_tensor_eye(size_t n)
{
    if (n == 0)
        return NULL;
    size_t shape[2] = {n, n};
    pg_tensor *t = pg_tensor_new(2, shape);
    if (!t)
        return NULL;
    for (size_t i = 0; i < n; i++)
        t->data[i * n + i] = 1.0f;
    return t;
}

pg_tensor *pg_tensor_clone(const pg_tensor *t)
{
    assert(t && t->data);
    return pg_tensor_from_data(t->ndim, t->shape, t->data);
}

static size_t flat_index(const pg_tensor *t, const size_t *idx)
{
    size_t off = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        assert(idx[i] < t->shape[i]);
        off += idx[i] * t->stride[i];
    }
    return off;
}

float pg_tensor_get(const pg_tensor *t, const size_t *idx)
{
    assert(t && t->data && idx);
    return t->data[flat_index(t, idx)];
}

void pg_tensor_set(pg_tensor *t, const size_t *idx, float value)
{
    assert(t && t->data && idx);
    t->data[flat_index(t, idx)] = value;
}

void pg_tensor_fill(pg_tensor *t, float value)
{
    assert(t && t->data);
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] = value;
}

void pg_tensor_copy_from(pg_tensor *dst, const pg_tensor *src)
{
    assert(dst && src && dst->data && src->data);
    assert(dst->ndim == src->ndim);
    assert(memcmp(dst->shape, src->shape, dst->ndim * sizeof(size_t)) == 0);
    memcpy(dst->data, src->data, src->numel * sizeof(float));
}

bool pg_tensor_reshape(pg_tensor *t, size_t ndim, const size_t *shape)
{
    if (!valid_shape(ndim, shape))
        return false;
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++)
        numel *= shape[i];
    if (numel != t->numel)
        return false;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, t->shape, t->stride);
    return true;
}

size_t pg_shape_numel(size_t ndim, const size_t *shape)
{
    if (!valid_shape(ndim, shape))
        return 0;
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++)
        numel *= shape[i];
    return numel;
}

bool pg_shape_equal(size_t ndim_a, const size_t *shape_a, size_t ndim_b, const size_t *shape_b)
{
    if (ndim_a != ndim_b)
        return false;
    for (size_t i = 0; i < ndim_a; i++)
        if (shape_a[i] != shape_b[i])
            return false;
    return true;
}

bool pg_tensor_allclose(const pg_tensor *a, const pg_tensor *b, float rtol, float atol)
{
    assert(a && b && a->data && b->data);
    if (!pg_shape_equal(a->ndim, a->shape, b->ndim, b->shape))
        return false;
    for (size_t i = 0; i < a->numel; i++) {
        float diff = fabsf(a->data[i] - b->data[i]);
        if (!(diff <= atol + rtol * fabsf(b->data[i])))
            return false;
    }
    return true;
}

static void print_rec(const pg_tensor *t, size_t dim, size_t offset, FILE *out)
{
    fputc('[', out);
    if (dim == t->ndim - 1) {
        for (size_t i = 0; i < t->shape[dim]; i++)
            fprintf(out, i ? ", %g" : "%g", t->data[offset + i]);
    } else {
        for (size_t i = 0; i < t->shape[dim]; i++) {
            if (i)
                fprintf(out, ",\n%*s", (int)(dim + 1), "");
            print_rec(t, dim + 1, offset + i * t->stride[dim], out);
        }
    }
    fputc(']', out);
}

void pg_tensor_print(const pg_tensor *t, FILE *out)
{
    assert(t && t->data && out);
    print_rec(t, 0, 0, out);
    fputc('\n', out);
}
