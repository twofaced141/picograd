#include "matmul.h"

#include "../backend/backend.h"
#include "common.h"
#include "elementwise.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void scale_(pg_tensor *t, float s)
{
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] *= s;
}

static pg_tensor *permute_copy(const pg_tensor *t, const size_t *order)
{
    size_t shape[PG_MAX_NDIM];
    for (size_t j = 0; j < t->ndim; j++) {
        assert(order[j] < t->ndim);
        shape[j] = t->shape[order[j]];
    }
    pg_tensor *out = pg_tensor_empty(t->ndim, shape);
    if (!out)
        return NULL;

    size_t midx[PG_MAX_NDIM] = {0};
    for (size_t p = 0; p < out->numel; p++) {
        size_t src = 0;
        for (size_t d = 0; d < t->ndim; d++)
            src += midx[d] * t->stride[order[d]];
        out->data[p] = t->data[src];
        pg_odometer_next(midx, out->shape, out->ndim);
    }
    return out;
}

static bool prod_dims(const pg_tensor *t, const size_t *dims, size_t ndims, size_t *out)
{
    size_t p = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (dims[i] >= t->ndim)
            return false;
        p *= t->shape[dims[i]];
    }
    *out = p;
    return true;
}

static pg_tensor *try_matmul_gpu(const pg_tensor *a, const pg_tensor *b)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;
    /* overflow guard before device copy */
    if (a->numel > UINT_MAX || b->numel > UINT_MAX)
        return NULL;

    size_t ra = a->ndim, rb = b->ndim;
    bool av = ra == 1, bv = rb == 1;
    if (!bv && a->shape[ra - 1] != b->shape[rb - 2])
        return NULL;
    if (a->stride[ra - 1] != 1)
        return NULL;
    if (!bv && b->stride[rb - 1] != 1)
        return NULL;

    size_t am = av ? 1 : a->shape[ra - 2];
    size_t ak = a->shape[ra - 1];
    size_t bn = bv ? 1 : b->shape[rb - 1];
    if (am > UINT_MAX || ak > UINT_MAX || bn > UINT_MAX)
        return NULL;

    size_t abn = av ? 0 : ra - 2;
    size_t bbn = bv ? 0 : rb - 2;

    size_t bnd;
    const size_t *bshape;
    if (abn == 0 && bbn == 0) {
        bnd = 0;
        bshape = NULL;
    } else if (abn == 0) {
        bnd = bbn;
        bshape = b->shape;
    } else if (bbn == 0) {
        bnd = abn;
        bshape = a->shape;
    } else {
        if (abn != bbn) return NULL;
        if (memcmp(a->shape, b->shape, abn * sizeof(size_t)) != 0) return NULL;
        bnd = abn;
        bshape = a->shape;
    }

    size_t nbatch = 1;
    for (size_t d = 0; d < bnd; d++) {
        if (bshape[d] > UINT_MAX) return NULL;
        if (nbatch > SIZE_MAX / bshape[d]) return NULL;
        nbatch *= bshape[d];
    }

    size_t rshape[PG_MAX_NDIM], rndim = 0;
    for (size_t d = 0; d < bnd; d++)
        rshape[rndim++] = bshape[d];
    if (!av)
        rshape[rndim++] = am;
    if (!bv)
        rshape[rndim++] = bn;
    if (rndim == 0)
        rshape[rndim++] = 1;

    pg_tensor *out = pg_tensor_empty(rndim, rshape);
    if (!out)
        return NULL;
    if (out->numel > UINT_MAX) {
        pg_tensor_free(out);
        return NULL;
    }

    /* allocate device buffers for whole tensors */
    size_t bytes_a = a->numel * sizeof(float);
    size_t bytes_b = b->numel * sizeof(float);
    size_t bytes_out = out->numel * sizeof(float);
    pg_dev_buf da = pg_dev_buf_new(bytes_a ? bytes_a : 1);
    pg_dev_buf db = pg_dev_buf_new(bytes_b ? bytes_b : 1);
    pg_dev_buf dc = pg_dev_buf_new(bytes_out ? bytes_out : 1);
    if (!da.ptr || !db.ptr || !dc.ptr) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da.ptr, a->data, bytes_a) != PG_OK ||
        pg_copy_h2d(db.ptr, b->data, bytes_b) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }

    size_t sa_bat[PG_MAX_NDIM] = {0};
    size_t sb_bat[PG_MAX_NDIM] = {0};
    for (size_t d = 0; d < bnd; d++) {
        if (abn > 0) sa_bat[d] = a->stride[d];
        if (bbn > 0) sb_bat[d] = b->stride[d];
    }

    size_t midx[PG_MAX_NDIM] = {0};
    size_t oa = 0, ob = 0;
    size_t lda_a = av ? a->stride[ra - 1] : a->stride[ra - 2];
    size_t ldb_b = bv ? b->stride[rb - 1] : b->stride[rb - 2];
    for (size_t s = 0; s < nbatch; s++) {
        pg_gemm(am, bn, ak,
                da.ptr + oa, lda_a,
                db.ptr + ob, ldb_b,
                dc.ptr + s * am * bn, bn);
        for (size_t d = bnd; d-- > 0;) {
            midx[d]++;
            oa += sa_bat[d];
            ob += sb_bat[d];
            if (midx[d] < bshape[d])
                break;
            midx[d] = 0;
            oa -= sa_bat[d] * bshape[d];
            ob -= sb_bat[d] * bshape[d];
        }
    }

    if (pg_dev_sync() != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data, dc.ptr, bytes_out) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
    return out;
}

pg_tensor *pg_matmul(const pg_tensor *a, const pg_tensor *b)
{
    assert(a && b && a->data && b->data);
    assert(a->ndim >= 1 && b->ndim >= 1);

    pg_tensor *g = try_matmul_gpu(a, b);
    if (g) return g;

    size_t ra = a->ndim, rb = b->ndim;
    bool av = ra == 1, bv = rb == 1;
    if (!bv)
        assert(a->shape[ra - 1] == b->shape[rb - 2]);
    assert(a->stride[ra - 1] == 1);
    assert(bv || b->stride[rb - 1] == 1);

    size_t am = av ? 1 : a->shape[ra - 2];
    size_t ak = a->shape[ra - 1];
    size_t bn = bv ? 1 : b->shape[rb - 1];

    size_t abn = av ? 0 : ra - 2;
    size_t bbn = bv ? 0 : rb - 2;

    size_t bnd;
    const size_t *bshape;
    if (abn == 0 && bbn == 0) {
        bnd = 0;
        bshape = NULL;
    } else if (abn == 0) {
        bnd = bbn;
        bshape = b->shape;
    } else if (bbn == 0) {
        bnd = abn;
        bshape = a->shape;
    } else {
        assert(abn == bbn);
        assert(memcmp(a->shape, b->shape, abn * sizeof(size_t)) == 0);
        bnd = abn;
        bshape = a->shape;
    }

    size_t nbatch = 1;
    for (size_t d = 0; d < bnd; d++)
        nbatch *= bshape[d];

    size_t rshape[PG_MAX_NDIM], rndim = 0;
    for (size_t d = 0; d < bnd; d++)
        rshape[rndim++] = bshape[d];
    if (!av)
        rshape[rndim++] = am;
    if (!bv)
        rshape[rndim++] = bn;
    if (rndim == 0)
        rshape[rndim++] = 1;

    pg_tensor *out = pg_tensor_empty(rndim, rshape);
    if (!out)
        return NULL;

    size_t sa_bat[PG_MAX_NDIM] = {0};
    size_t sb_bat[PG_MAX_NDIM] = {0};
    for (size_t d = 0; d < bnd; d++) {
        if (abn > 0)
            sa_bat[d] = a->stride[d];
        if (bbn > 0)
            sb_bat[d] = b->stride[d];
    }

    size_t midx[PG_MAX_NDIM] = {0};
    size_t oa = 0, ob = 0;
    for (size_t s = 0; s < nbatch; s++) {
        pg_gemm(am, bn, ak,
                a->data + oa, av ? a->stride[ra - 1] : a->stride[ra - 2],
                b->data + ob, bv ? b->stride[rb - 1] : b->stride[rb - 2],
                out->data + s * am * bn, bn);
        for (size_t d = bnd; d-- > 0;) {
            midx[d]++;
            oa += sa_bat[d];
            ob += sb_bat[d];
            if (midx[d] < bshape[d])
                break;
            midx[d] = 0;
            oa -= sa_bat[d] * bshape[d];
            ob -= sb_bat[d] * bshape[d];
        }
    }
    return out;
}

static pg_tensor *try_bmm_gpu(const pg_tensor *a, const pg_tensor *b)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;
    if (a->numel > UINT_MAX || b->numel > UINT_MAX)
        return NULL;
    size_t batch = a->shape[0];
    size_t m = a->shape[1];
    size_t k = a->shape[2];
    size_t n = b->shape[2];
    if (m > UINT_MAX || n > UINT_MAX || k > UINT_MAX || batch > UINT_MAX)
        return NULL;
    size_t shape[3] = {batch, m, n};
    pg_tensor *out = pg_tensor_empty(3, shape);
    if (!out) return NULL;
    if (out->numel > UINT_MAX) { pg_tensor_free(out); return NULL; }

    size_t bytes_a = a->numel * sizeof(float);
    size_t bytes_b = b->numel * sizeof(float);
    size_t bytes_out = out->numel * sizeof(float);
    pg_dev_buf da = pg_dev_buf_new(bytes_a ? bytes_a : 1);
    pg_dev_buf db = pg_dev_buf_new(bytes_b ? bytes_b : 1);
    pg_dev_buf dc = pg_dev_buf_new(bytes_out ? bytes_out : 1);
    if (!da.ptr || !db.ptr || !dc.ptr) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da.ptr, a->data, bytes_a) != PG_OK ||
        pg_copy_h2d(db.ptr, b->data, bytes_b) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    for (size_t s = 0; s < batch; s++)
        pg_gemm(m, n, k,
                da.ptr + s * m * k, k,
                db.ptr + s * k * n, n,
                dc.ptr + s * m * n, n);
    if (pg_dev_sync() != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data, dc.ptr, bytes_out) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
    return out;
}

pg_tensor *pg_bmm(const pg_tensor *a, const pg_tensor *b)
{
    assert(a && b && a->data && b->data);
    assert(a->ndim == 3 && b->ndim == 3);
    assert(a->shape[0] == b->shape[0]);
    assert(a->shape[2] == b->shape[1]);

    pg_tensor *g = try_bmm_gpu(a, b);
    if (g) return g;

    size_t batch = a->shape[0];
    size_t m = a->shape[1];
    size_t k = a->shape[2];
    size_t n = b->shape[2];

    size_t shape[3] = {batch, m, n};
    pg_tensor *out = pg_tensor_empty(3, shape);
    if (!out)
        return NULL;

    for (size_t s = 0; s < batch; s++)
        pg_gemm(m, n, k,
                a->data + s * m * k, k,
                b->data + s * k * n, n,
                out->data + s * m * n, n);
    return out;
}

pg_tensor *pg_addmm(const pg_tensor *input, const pg_tensor *m1, const pg_tensor *m2,
                    float alpha, float beta)
{
    assert(input && m1 && m2 && input->data);
    assert(m1->ndim == 2 && m2->ndim == 2);
    assert(m1->shape[1] == m2->shape[0]);

    size_t m = m1->shape[0];
    size_t n = m2->shape[1];

    pg_tensor *tmp = pg_matmul(m1, m2);
    if (!tmp)
        return NULL;
    if (alpha != 1.0f)
        scale_(tmp, alpha);
    if (beta == 0.0f)
        return tmp;

    size_t zshape[2] = {m, n};
    pg_tensor *zeros = pg_tensor_zeros(2, zshape);
    pg_tensor *base = zeros ? pg_add(zeros, input) : NULL;
    pg_tensor_free(zeros);
    if (!base) {
        pg_tensor_free(tmp);
        return NULL;
    }
    if (beta != 1.0f)
        scale_(base, beta);

    pg_tensor *res = pg_add(base, tmp);
    pg_tensor_free(base);
    pg_tensor_free(tmp);
    return res;
}

pg_tensor *pg_tensordot(const pg_tensor *a, const pg_tensor *b,
                        size_t ndims, const size_t *axes_a, const size_t *axes_b)
{
    assert(a && b && a->data && b->data && axes_a && axes_b);
    assert(ndims >= 1 && ndims <= a->ndim && ndims <= b->ndim);

    size_t free_a = a->ndim - ndims;
    size_t rest_b = b->ndim - ndims;
    assert(free_a + rest_b <= PG_MAX_NDIM);

    for (size_t i = 0; i < ndims; i++) {
        assert(axes_a[i] < a->ndim && axes_b[i] < b->ndim);
        assert(a->shape[axes_a[i]] == b->shape[axes_b[i]]);
    }

    size_t pa[PG_MAX_NDIM], pb[PG_MAX_NDIM];
    size_t w = 0;
    for (size_t d = 0; d < a->ndim; d++) {
        bool contracted = false;
        for (size_t i = 0; i < ndims; i++)
            if (axes_a[i] == d)
                contracted = true;
        if (!contracted)
            pa[w++] = d;
    }
    for (size_t i = 0; i < ndims; i++)
        pa[w++] = axes_a[i];

    w = 0;
    for (size_t i = 0; i < ndims; i++)
        pb[w++] = axes_b[i];
    for (size_t d = 0; d < b->ndim; d++) {
        bool contracted = false;
        for (size_t i = 0; i < ndims; i++)
            if (axes_b[i] == d)
                contracted = true;
        if (!contracted)
            pb[w++] = d;
    }

    size_t mdim, kdim, ndim_n;
    if (!prod_dims(a, pa, free_a, &mdim))
        return NULL;
    if (!prod_dims(a, pa + free_a, ndims, &kdim))
        return NULL;
    if (!prod_dims(b, pb + ndims, rest_b, &ndim_n))
        return NULL;

    pg_tensor *ra = permute_copy(a, pa);
    pg_tensor *rb = permute_copy(b, pb);
    if (!ra || !rb) {
        pg_tensor_free(ra);
        pg_tensor_free(rb);
        return NULL;
    }

    bool oka = pg_tensor_reshape(ra, 2, (size_t[]){mdim, kdim});
    bool okb = pg_tensor_reshape(rb, 2, (size_t[]){kdim, ndim_n});
    if (!oka || !okb) {
        pg_tensor_free(ra);
        pg_tensor_free(rb);
        return NULL;
    }

    pg_tensor *rc = pg_matmul(ra, rb);
    pg_tensor_free(ra);
    pg_tensor_free(rb);
    if (!rc)
        return NULL;

    size_t fshape[PG_MAX_NDIM];
    size_t j = 0;
    for (size_t d = 0; d < free_a; d++)
        fshape[j++] = a->shape[pa[d]];
    for (size_t d = ndims; d < b->ndim; d++)
        fshape[j++] = b->shape[pb[d]];
    if (j == 0)
        fshape[j++] = 1;

    bool ok = pg_tensor_reshape(rc, j, fshape);
    assert(ok);
    (void)ok;
    return rc;
}
