#include "scan.h"

#include "common.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float v;
    size_t i;
} pair;

static void merge_pairs(const pair *a, size_t na, const pair *b, size_t nb, pair *dst, bool desc)
{
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        bool take_left = desc ? a[i].v >= b[j].v : a[i].v <= b[j].v;
        dst[k++] = take_left ? a[i++] : b[j++];
    }
    while (i < na)
        dst[k++] = a[i++];
    while (j < nb)
        dst[k++] = b[j++];
}

static int sort_pairs(pair *v, size_t n, bool desc)
{
    if (n < 2)
        return 1;
    pair *tmp = malloc(n * sizeof(pair));
    if (!tmp)
        return 0;
    for (size_t w = 1; w < n; w *= 2) {
        for (size_t lo = 0; lo + w < n; lo += 2 * w) {
            size_t mid = lo + w;
            size_t hi = lo + 2 * w < n ? lo + 2 * w : n;
            merge_pairs(v + lo, mid - lo, v + mid, hi - mid, tmp + lo, desc);
            memcpy(v + lo, tmp + lo, (hi - lo) * sizeof(pair));
        }
    }
    free(tmp);
    return 1;
}

static void advance_skip(size_t *idx, const size_t *shape, size_t ndim, size_t axis)
{
    for (size_t d = ndim; d-- > 0;) {
        if (d == axis)
            continue;
        idx[d]++;
        if (idx[d] < shape[d])
            return;
        idx[d] = 0;
    }
}

static size_t line_base(const pg_tensor *t, const size_t *idx, size_t axis)
{
    size_t base = 0;
    for (size_t d = 0; d < t->ndim; d++)
        if (d != axis)
            base += idx[d] * t->stride[d];
    return base;
}

static pg_tensor *cumulate(const pg_tensor *t, size_t axis, bool prod)
{
    assert(t && t->data && t->ndim >= 1 && axis < t->ndim);

    pg_tensor *out = pg_tensor_clone(t);
    if (!out)
        return NULL;

    size_t st = out->stride[axis];
    size_t dim = out->shape[axis];
    size_t lines = out->numel / dim;
    size_t midx[PG_MAX_NDIM] = {0};

    for (size_t l = 0; l < lines; l++) {
        size_t base = line_base(out, midx, axis);
        float acc = prod ? 1.0f : 0.0f;
        for (size_t v = 0; v < dim; v++) {
            size_t p = base + v * st;
            acc = prod ? acc * out->data[p] : acc + out->data[p];
            out->data[p] = acc;
        }
        advance_skip(midx, out->shape, out->ndim, axis);
    }
    return out;
}

pg_tensor *pg_cumsum(const pg_tensor *t, size_t axis)
{
    return cumulate(t, axis, false);
}

pg_tensor *pg_cumprod(const pg_tensor *t, size_t axis)
{
    return cumulate(t, axis, true);
}

static void fill_out_shape(const pg_tensor *t, size_t axis, bool keep, size_t k, size_t *shape, size_t *ondim)
{
    if (keep) {
        for (size_t d = 0; d < t->ndim; d++)
            shape[d] = d == axis ? k : t->shape[d];
        *ondim = t->ndim;
    } else {
        size_t j = 0;
        for (size_t d = 0; d < t->ndim; d++)
            if (d != axis)
                shape[j++] = t->shape[d];
        shape[j++] = k;
        *ondim = j;
    }
}

static pg_kv sort_topk_impl(const pg_tensor *t, size_t axis, size_t take, bool descending)
{
    pg_kv kv = {NULL, NULL};
    assert(t && t->data && t->ndim >= 1 && axis < t->ndim);
    assert(take >= 1 && take <= t->shape[axis]);

    size_t shape[PG_MAX_NDIM], ondim;
    fill_out_shape(t, axis, true, take, shape, &ondim);
    kv.values = pg_tensor_empty(ondim, shape);
    kv.indices = pg_tensor_empty(ondim, shape);
    if (!kv.values || !kv.indices)
        goto fail;

    size_t st = t->stride[axis];
    size_t dim = t->shape[axis];
    size_t lines = t->numel / dim;
    size_t midx[PG_MAX_NDIM] = {0};
    pair *buf = NULL;

    buf = malloc(dim * sizeof(pair));
    if (!buf)
        goto fail;

    for (size_t l = 0; l < lines; l++) {
        size_t base = line_base(t, midx, axis);
        for (size_t v = 0; v < dim; v++) {
            buf[v].v = t->data[base + v * st];
            buf[v].i = v;
        }
        sort_pairs(buf, dim, descending);
        size_t obase = l * take;
        for (size_t v = 0; v < take; v++) {
            kv.values->data[obase + v] = buf[v].v;
            kv.indices->data[obase + v] = (float)buf[v].i;
        }
        advance_skip(midx, t->shape, t->ndim, axis);
    }
    free(buf);
    return kv;

fail:
    free(buf);
    pg_tensor_free(kv.values);
    pg_tensor_free(kv.indices);
    kv.values = NULL;
    kv.indices = NULL;
    return kv;
}

pg_kv pg_sort(const pg_tensor *t, size_t axis, bool descending)
{
    return sort_topk_impl(t, axis, t ? t->shape[axis] : 0, descending);
}

pg_kv pg_topk(const pg_tensor *t, size_t axis, size_t k)
{
    return sort_topk_impl(t, axis, k, true);
}
