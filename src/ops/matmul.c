#include "matmul.h"

#include "../backend/backend.h"
#include "../core/convert.h"
#include "common.h"
#include "elementwise.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void scale_(pg_tensor *t, float s)
{
    // only for f32 tensors (accum)
    assert(t->dtype==PG_DTYPE_F32);
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] *= s;
}

static pg_tensor *permute_copy(const pg_tensor *t, const size_t *order)
{
    if (!t || !t->data_raw || !order) return NULL;
    size_t shape[PG_MAX_NDIM];
    for (size_t j = 0; j < t->ndim; j++) {
        if (order[j] >= t->ndim) return NULL;
        shape[j] = t->shape[order[j]];
    }
    pg_tensor *out = pg_tensor_empty_dtype(t->dtype, t->ndim, shape);
    if (!out) return NULL;

    size_t midx[PG_MAX_NDIM] = {0};
    size_t es = t->elem_size;
    for (size_t p = 0; p < out->numel; p++) {
        size_t src = 0;
        for (size_t d = 0; d < t->ndim; d++)
            src += midx[d] * t->stride[order[d]];
        memcpy((char*)out->data_raw + p*es, (char*)t->data_raw + src*es, es);
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
    if (a->numel > UINT_MAX || b->numel > UINT_MAX)
        return NULL;
    // only support f32 on gpu for now (future wmma will handle f16)
    if (a->dtype!=PG_DTYPE_F32 || b->dtype!=PG_DTYPE_F32) return NULL;

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

    size_t bytes_a = a->numel * a->elem_size;
    size_t bytes_b = b->numel * b->elem_size;
    size_t bytes_out = out->numel * out->elem_size;
    pg_dev_buf da = pg_dev_buf_new(bytes_a ? bytes_a : 1);
    pg_dev_buf db = pg_dev_buf_new(bytes_b ? bytes_b : 1);
    pg_dev_buf dc = pg_dev_buf_new(bytes_out ? bytes_out : 1);
    if (!da.ptr || !db.ptr || !dc.ptr) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da.ptr, a->data_raw, bytes_a) != PG_OK ||
        pg_copy_h2d(db.ptr, b->data_raw, bytes_b) != PG_OK) {
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
                (float*)da.ptr + oa, lda_a,
                (float*)db.ptr + ob, ldb_b,
                (float*)dc.ptr + s * am * bn, bn);
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
    if (pg_copy_d2h(out->data_raw, dc.ptr, bytes_out) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
    return out;
}

pg_tensor *pg_matmul(const pg_tensor *a, const pg_tensor *b)
{
    if (!a || !b || !a->data_raw || !b->data_raw) return NULL;
    if (a->ndim < 1 || b->ndim < 1) return NULL;
    // dtype check: allow both f32, both f16, both bf16. Mixed f16/bf16 not allowed.
    if (a->dtype != b->dtype) return NULL;

    pg_tensor *g = try_matmul_gpu(a, b);
    if (g) return g;

    size_t ra = a->ndim, rb = b->ndim;
    bool av = ra == 1, bv = rb == 1;
    if (!bv && a->shape[ra - 1] != b->shape[rb - 2]) return NULL;
    if (a->stride[ra - 1] != 1) return NULL;
    if (!bv && b->stride[rb - 1] != 1) return NULL;

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
        if (abn != bbn) return NULL;
        if (memcmp(a->shape, b->shape, abn * sizeof(size_t)) != 0) return NULL;
        bnd = abn;
        bshape = a->shape;
    }

    size_t nbatch = 1;
    for (size_t d = 0; d < bnd; d++) {
        if (bshape[d]==0) return NULL;
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

    // output always F32 for mixed precision (f32 accum)
    pg_tensor *out = pg_tensor_empty_dtype(PG_DTYPE_F32, rndim, rshape);
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
    pg_dtype dt = a->dtype;
    for (size_t s = 0; s < nbatch; s++) {
        size_t lda_a = av ? a->stride[ra - 1] : a->stride[ra - 2];
        size_t ldb_b = bv ? b->stride[rb - 1] : b->stride[rb - 2];
        const void *pa = (char*)a->data_raw + oa * a->elem_size;
        const void *pb = (char*)b->data_raw + ob * b->elem_size;
        float *pc = (float*)out->data_raw + s * am * bn;
        // ld* are in elements (not bytes) per spec
        if (dt==PG_DTYPE_F32) pg_gemm(am, bn, ak, (const float*)pa, lda_a, (const float*)pb, ldb_b, pc, bn);
        else pg_gemm_ex(dt, am, bn, ak, pa, lda_a, pb, ldb_b, pc, bn);
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
    if (a->dtype!=PG_DTYPE_F32 || b->dtype!=PG_DTYPE_F32) return NULL;
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

    size_t bytes_a = a->numel * a->elem_size;
    size_t bytes_b = b->numel * b->elem_size;
    size_t bytes_out = out->numel * out->elem_size;
    pg_dev_buf da = pg_dev_buf_new(bytes_a ? bytes_a : 1);
    pg_dev_buf db = pg_dev_buf_new(bytes_b ? bytes_b : 1);
    pg_dev_buf dc = pg_dev_buf_new(bytes_out ? bytes_out : 1);
    if (!da.ptr || !db.ptr || !dc.ptr) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da.ptr, a->data_raw, bytes_a) != PG_OK ||
        pg_copy_h2d(db.ptr, b->data_raw, bytes_b) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    for (size_t s = 0; s < batch; s++)
        pg_gemm(m, n, k,
                (float*)da.ptr + s * m * k, k,
                (float*)db.ptr + s * k * n, n,
                (float*)dc.ptr + s * m * n, n);
    if (pg_dev_sync() != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data_raw, dc.ptr, bytes_out) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
    return out;
}

pg_tensor *pg_bmm(const pg_tensor *a, const pg_tensor *b)
{
    if (!a || !b || !a->data_raw || !b->data_raw) return NULL;
    if (a->ndim != 3 || b->ndim != 3) return NULL;
    if (a->shape[0] != b->shape[0]) return NULL;
    if (a->shape[2] != b->shape[1]) return NULL;
    if (a->dtype != b->dtype) return NULL;

    pg_tensor *g = try_bmm_gpu(a, b);
    if (g) return g;

    size_t batch = a->shape[0];
    size_t m = a->shape[1];
    size_t k = a->shape[2];
    size_t n = b->shape[2];

    size_t shape[3] = {batch, m, n};
    pg_tensor *out = pg_tensor_empty_dtype(PG_DTYPE_F32, 3, shape);
    if (!out)
        return NULL;
    pg_dtype dt=a->dtype;
    for (size_t s = 0; s < batch; s++) {
        const void *pa = (char*)a->data_raw + s * m * k * a->elem_size;
        const void *pb = (char*)b->data_raw + s * k * n * b->elem_size;
        float *pc = (float*)out->data_raw + s * m * n;
        if (dt==PG_DTYPE_F32) pg_gemm(m,n,k,(const float*)pa,k,(const float*)pb,n,pc,n);
        else pg_gemm_ex(dt,m,n,k,pa,k,pb,n,pc,n);
    }
    return out;
}

pg_tensor *pg_addmm(const pg_tensor *input, const pg_tensor *m1, const pg_tensor *m2,
                    float alpha, float beta)
{
    if (!input || !m1 || !m2 || !input->data_raw || !m1->data_raw || !m2->data_raw) return NULL;
    if (m1->ndim != 2 || m2->ndim != 2) return NULL;
    if (m1->shape[1] != m2->shape[0]) return NULL;
    if (m1->dtype != m2->dtype) return NULL;

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
    if (!a || !b || !a->data_raw || !b->data_raw || !axes_a || !axes_b) return NULL;
    if (ndims < 1 || ndims > a->ndim || ndims > b->ndim) return NULL;
    if (a->dtype != b->dtype) return NULL;

    size_t free_a = a->ndim - ndims;
    size_t rest_b = b->ndim - ndims;
    if (free_a + rest_b > PG_MAX_NDIM) return NULL;

    for (size_t i = 0; i < ndims; i++) {
        if (axes_a[i] >= a->ndim || axes_b[i] >= b->ndim) return NULL;
        if (a->shape[axes_a[i]] != b->shape[axes_b[i]]) return NULL;
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
    if (!ok) { pg_tensor_free(rc); return NULL; }
    return rc;
}
