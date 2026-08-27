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
    assert(t && indices && t->data && indices->data);
    assert(indices->ndim == t->ndim && axis < t->ndim);
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
    assert(t && indices && src && t->data && indices->data && src->data);
    assert(indices->ndim == t->ndim && axis < t->ndim);
    assert(indices->ndim == src->ndim);
    assert(memcmp(indices->shape, src->shape, indices->ndim * sizeof(size_t)) == 0);
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

pg_tensor *pg_index_select(const pg_tensor *t, size_t axis, const pg_tensor *indices)
{
    assert(t && indices && t->data && indices->data);
    assert(t->ndim >= 1 && axis < t->ndim && indices->ndim == 1);
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
    assert(t && mask && t->data && mask->data);
    assert(pg_shape_equal(t->ndim, t->shape, mask->ndim, mask->shape));

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
