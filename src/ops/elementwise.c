#include "elementwise.h"

#include "common.h"
#include "../backend/backend.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float fadd(float x, float y) { return x + y; }
static float fsub(float x, float y) { return x - y; }
static float fmul(float x, float y) { return x * y; }
static float fdiv(float x, float y) { return x / y; }
static float fpow(float x, float y) { return powf(x, y); }

static pg_tensor *try_bin_gpu(const pg_tensor *a, const pg_tensor *b, int bin_op)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;

    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t shape[PG_MAX_NDIM];
    if (!pg_bcast_shape(a, b, &ndim, shape))
        return NULL;
    if (ndim > PG_MAX_OP_NDIM)
        return NULL;

    size_t sa[PG_MAX_NDIM], sb[PG_MAX_NDIM];
    if (!pg_bcast_strides(a, ndim, shape, sa) || !pg_bcast_strides(b, ndim, shape, sb))
        return NULL;

    pg_tensor *out = pg_tensor_new(ndim, shape);
    if (!out)
        return NULL;

    if (out->numel > UINT_MAX || a->numel > UINT_MAX || b->numel > UINT_MAX) {
        pg_tensor_free(out);
        return NULL;
    }

    /* Build kernel args */
    pg_k_bin_args kargs;
    memset(&kargs, 0, sizeof(kargs));
    kargs.ndim = (unsigned)ndim;
    kargs.numel = (unsigned)out->numel;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] > UINT_MAX || sa[i] > UINT_MAX || sb[i] > UINT_MAX) {
            pg_tensor_free(out);
            return NULL;
        }
        kargs.shape[i] = (unsigned)shape[i];
        kargs.sa[i] = (unsigned)sa[i];
        kargs.sb[i] = (unsigned)sb[i];
    }

    size_t bytes_a = a->numel * sizeof(float);
    size_t bytes_b = b->numel * sizeof(float);
    size_t bytes_out = out->numel * sizeof(float);

    float *da = pg_dev_malloc(bytes_a ? bytes_a : 1);
    float *db = pg_dev_malloc(bytes_b ? bytes_b : 1);
    float *dc = pg_dev_malloc(bytes_out ? bytes_out : 1);
    if (!da || !db || !dc) {
        if (da) pg_dev_free(da);
        if (db) pg_dev_free(db);
        if (dc) pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }

    if (pg_copy_h2d(da, a->data, bytes_a) != PG_OK ||
        pg_copy_h2d(db, b->data, bytes_b) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(db);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }

    pg_status st = pg_op_bin(dc, da, db, out->numel, bin_op, &kargs);
    if (st != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(db);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_dev_sync() != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(db);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data, dc, bytes_out) != PG_OK) {
        pg_dev_free(da);
        pg_dev_free(db);
        pg_dev_free(dc);
        pg_tensor_free(out);
        return NULL;
    }

    pg_dev_free(da);
    pg_dev_free(db);
    pg_dev_free(dc);
    return out;
}

static pg_tensor *bcast_binary(const pg_tensor *a, const pg_tensor *b, float (*f)(float, float))
{
    assert(a && b && a->data && b->data);

    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t shape[PG_MAX_NDIM];
    if (!pg_bcast_shape(a, b, &ndim, shape))
        return NULL;

    size_t sa[PG_MAX_NDIM], sb[PG_MAX_NDIM];
    if (!pg_bcast_strides(a, ndim, shape, sa) || !pg_bcast_strides(b, ndim, shape, sb))
        return NULL;

    pg_tensor *out = pg_tensor_new(ndim, shape);
    if (!out)
        return NULL;

    size_t idx[PG_MAX_NDIM] = {0};
    size_t oa = 0, ob = 0;
    for (size_t p = 0; p < out->numel; p++) {
        out->data[p] = f(a->data[oa], b->data[ob]);
        for (size_t d = out->ndim; d-- > 0;) {
            idx[d]++;
            oa += sa[d];
            ob += sb[d];
            if (idx[d] < out->shape[d])
                break;
            idx[d] = 0;
            oa -= sa[d] * out->shape[d];
            ob -= sb[d] * out->shape[d];
        }
    }
    return out;
}

pg_tensor *pg_add(const pg_tensor *a, const pg_tensor *b)
{
    pg_tensor *g = try_bin_gpu(a, b, PG_BIN_ADD);
    if (g)
        return g;
    return bcast_binary(a, b, fadd);
}

pg_tensor *pg_sub(const pg_tensor *a, const pg_tensor *b)
{
    pg_tensor *g = try_bin_gpu(a, b, PG_BIN_SUB);
    if (g)
        return g;
    return bcast_binary(a, b, fsub);
}

pg_tensor *pg_mul(const pg_tensor *a, const pg_tensor *b)
{
    pg_tensor *g = try_bin_gpu(a, b, PG_BIN_MUL);
    if (g)
        return g;
    return bcast_binary(a, b, fmul);
}

pg_tensor *pg_div(const pg_tensor *a, const pg_tensor *b)
{
    pg_tensor *g = try_bin_gpu(a, b, PG_BIN_DIV);
    if (g)
        return g;
    return bcast_binary(a, b, fdiv);
}

pg_tensor *pg_pow(const pg_tensor *a, const pg_tensor *b)
{
    return bcast_binary(a, b, fpow);
}

static pg_tensor *try_map_gpu(const pg_tensor *a, int map_op)
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
    pg_status st = pg_op_map(dc, da, a->numel, map_op);
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

static pg_tensor *map1(const pg_tensor *a, float (*f)(float))
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    for (size_t i = 0; i < r->numel; i++)
        r->data[i] = f(r->data[i]);
    return r;
}

pg_tensor *pg_exp(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_EXP);
    if (g) return g;
    return map1(a, expf);
}
pg_tensor *pg_log(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_LOG);
    if (g) return g;
    return map1(a, logf);
}
pg_tensor *pg_sin(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_SIN);
    if (g) return g;
    return map1(a, sinf);
}
pg_tensor *pg_cos(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_COS);
    if (g) return g;
    return map1(a, cosf);
}
pg_tensor *pg_erf(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_ERF);
    if (g) return g;
    return map1(a, erff);
}

static float fneg(float x) { return -x; }
static float fsqrt(float x) { return sqrtf(x); }

pg_tensor *pg_neg(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_NEG);
    if (g) return g;
    return map1(a, fneg);
}
pg_tensor *pg_abs(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_ABS);
    if (g) return g;
    return map1(a, fabsf);
}
pg_tensor *pg_sqrt(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu(a, PG_MAP_SQRT);
    if (g) return g;
    return map1(a, fsqrt);
}

pg_tensor *pg_clamp(const pg_tensor *a, float lo, float hi)
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    for (size_t i = 0; i < r->numel; i++) {
        float v = r->data[i];
        r->data[i] = v < lo ? lo : (v > hi ? hi : v);
    }
    return r;
}
