#include "index.h"

#include "common.h"
#include <assert.h>
#include <math.h>
#include <string.h>

static bool indices_valid(const pg_tensor *idx, size_t limit)
{
    for (size_t i = 0; i < idx->numel; i++) {
        float v = idx->data[i];
        if (!(v >= 0.0f) || v >= (float)limit || v != floorf(v))
            return false;
    }
    return true;
}

static void advance(size_t *midx, const size_t *shape, size_t ndim)
{
    pg_odometer_next(midx, shape, ndim);
}

pg_tensor *pg_gather(const pg_tensor *t, size_t axis, const pg_tensor *indices)
{
    if (!t || !indices || !t->data || !indices->data) return NULL;
    if (indices->ndim != t->ndim || axis >= t->ndim) return NULL;
    if (!indices_valid(indices, t->shape[axis]))
        return NULL;

    pg_tensor *out = pg_tensor_empty(indices->ndim, indices->shape);
    if (!out)
        return NULL;

    size_t midx[PG_MAX_NDIM] = {0};
    for (size_t p = 0; p < out->numel; p++) {
        size_t j = (size_t)indices->data[p];
        size_t src = j * t->stride[axis];
        for (size_t d = 0; d < t->ndim; d++)
            if (d != axis)
                src += midx[d] * t->stride[d];
        out->data[p] = t->data[src];
        advance(midx, indices->shape, indices->ndim);
    }
    return out;
}

pg_tensor *pg_scatter(const pg_tensor *t, size_t axis, const pg_tensor *indices, const pg_tensor *src)
{
    if (!t || !indices || !src || !t->data || !indices->data || !src->data) return NULL;
    if (indices->ndim != t->ndim || axis >= t->ndim) return NULL;
    if (indices->ndim != src->ndim) return NULL;
    if (memcmp(indices->shape, src->shape, indices->ndim * sizeof(size_t)) != 0) return NULL;
    if (!indices_valid(indices, t->shape[axis]))
        return NULL;

    pg_tensor *out = pg_tensor_clone(t);
    if (!out)
        return NULL;

    size_t midx[PG_MAX_NDIM] = {0};
    for (size_t p = 0; p < indices->numel; p++) {
        size_t j = (size_t)indices->data[p];
        size_t dst = j * out->stride[axis];
        for (size_t d = 0; d < t->ndim; d++)
            if (d != axis)
                dst += midx[d] * out->stride[d];
        out->data[dst] = src->data[p];
        advance(midx, indices->shape, indices->ndim);
    }
    return out;
}

pg_tensor *pg_embedding(const pg_tensor *weight, const pg_tensor *indices)
{
    if (!weight || !indices || !weight->data || !indices->data) return NULL;
    if (weight->ndim != 2) return NULL;
    size_t V = weight->shape[0], E = weight->shape[1];

    for (size_t i = 0; i < indices->numel; i++) {
        float v = indices->data[i];
        if (!(v >= 0.0f && v < (float)V && v == floorf(v)))
            return NULL;
    }

    size_t shape[PG_MAX_NDIM];
    if (indices->ndim + 1 > PG_MAX_NDIM) return NULL;
    memcpy(shape, indices->shape, indices->ndim * sizeof(size_t));
    shape[indices->ndim] = E;
    size_t ndim = indices->ndim + 1;

    pg_tensor *out = pg_tensor_empty(ndim, shape);
    if (!out)
        return NULL;

    for (size_t i = 0; i < indices->numel; i++) {
        size_t row = (size_t)indices->data[i];
        memcpy(out->data + i * E, weight->data + row * E, E * sizeof(float));
    }
    return out;
}

pg_tensor *pg_index_select(const pg_tensor *t, size_t axis, const pg_tensor *indices)
{
    if (!t || !indices || !t->data || !indices->data) return NULL;
    if (t->ndim < 1 || axis >= t->ndim || indices->ndim != 1) return NULL;
    if (!indices_valid(indices, t->shape[axis]))
        return NULL;

    size_t shape[PG_MAX_NDIM];
    memcpy(shape, t->shape, t->ndim * sizeof(size_t));
    shape[axis] = indices->numel;
    pg_tensor *out = pg_tensor_empty(t->ndim, shape);
    if (!out)
        return NULL;

    size_t midx[PG_MAX_NDIM] = {0};
    for (size_t p = 0; p < out->numel; p++) {
        size_t j = (size_t)indices->data[midx[axis]];
        size_t src = 0;
        for (size_t d = 0; d < t->ndim; d++)
            src += (d == axis ? j : midx[d]) * t->stride[d];
        out->data[p] = t->data[src];
        advance(midx, out->shape, out->ndim);
    }
    return out;
}

pg_tensor *pg_masked_select(const pg_tensor *t, const pg_tensor *mask)
{
    if (!t || !mask || !t->data || !mask->data) return NULL;
    if (!pg_shape_equal(t->ndim, t->shape, mask->ndim, mask->shape)) return NULL;

    size_t n = 0;
    for (size_t i = 0; i < t->numel; i++)
        if (mask->data[i] != 0.0f)
            n++;

    size_t shape[1] = {n};
    pg_tensor *out = pg_tensor_empty(1, shape);
    if (!out)
        return NULL;

    size_t w = 0;
    for (size_t i = 0; i < t->numel; i++)
        if (mask->data[i] != 0.0f)
            out->data[w++] = t->data[i];
    return out;
}
