#ifndef PICOGRAD_OPS_COMMON_H
#define PICOGRAD_OPS_COMMON_H

#include "../core/tensor.h"

static inline bool pg_bcast_shape(const pg_tensor *a, const pg_tensor *b, size_t *out_ndim, size_t *out_shape)
{
    if (!a || !b || !a->data || !b->data)
        return false;
    size_t nd = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (size_t d = 0; d < nd; d++) {
        size_t da = d < nd - a->ndim ? 1 : a->shape[d - (nd - a->ndim)];
        size_t db = d < nd - b->ndim ? 1 : b->shape[d - (nd - b->ndim)];
        if (da != db && da != 1 && db != 1)
            return false;
    }
    *out_ndim = nd;
    for (size_t d = 0; d < nd; d++) {
        size_t da = d < nd - a->ndim ? 1 : a->shape[d - (nd - a->ndim)];
        size_t db = d < nd - b->ndim ? 1 : b->shape[d - (nd - b->ndim)];
        out_shape[d] = da > db ? da : db;
    }
    return true;
}

static inline void pg_axis_split(size_t ndim, const size_t *shape, size_t axis,
                                 size_t *outer, size_t *len, size_t *inner)
{
    *outer = 1;
    *len = shape[axis];
    *inner = 1;
    for (size_t d = 0; d < axis; d++)
        *outer *= shape[d];
    for (size_t d = axis + 1; d < ndim; d++)
        *inner *= shape[d];
}

static inline bool pg_bcast_strides(const pg_tensor *t, size_t out_ndim, const size_t *out_shape, size_t *strides)
{
    if (t->ndim > out_ndim)
        return false;
    size_t off = out_ndim - t->ndim;
    for (size_t d = 0; d < off; d++)
        strides[d] = 0;
    for (size_t d = off, e = 0; e < t->ndim; d++, e++) {
        if (t->shape[e] == out_shape[d])
            strides[d] = t->stride[e];
        else if (t->shape[e] == 1)
            strides[d] = 0;
        else
            return false;
    }
    return true;
}

static inline void pg_odometer_next(size_t *idx, const size_t *shape, size_t ndim)
{
    for (size_t d = ndim; d-- > 0;) {
        idx[d]++;
        if (idx[d] < shape[d])
            return;
        idx[d] = 0;
    }
}

#endif
