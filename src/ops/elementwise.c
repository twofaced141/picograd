#include "elementwise.h"

#include "common.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

static float fadd(float x, float y) { return x + y; }
static float fsub(float x, float y) { return x - y; }
static float fmul(float x, float y) { return x * y; }
static float fdiv(float x, float y) { return x / y; }
static float fpow(float x, float y) { return powf(x, y); }

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
    return bcast_binary(a, b, fadd);
}

pg_tensor *pg_sub(const pg_tensor *a, const pg_tensor *b)
{
    return bcast_binary(a, b, fsub);
}

pg_tensor *pg_mul(const pg_tensor *a, const pg_tensor *b)
{
    return bcast_binary(a, b, fmul);
}

pg_tensor *pg_div(const pg_tensor *a, const pg_tensor *b)
{
    return bcast_binary(a, b, fdiv);
}

pg_tensor *pg_pow(const pg_tensor *a, const pg_tensor *b)
{
    return bcast_binary(a, b, fpow);
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

pg_tensor *pg_exp(const pg_tensor *a) { return map1(a, expf); }
pg_tensor *pg_log(const pg_tensor *a) { return map1(a, logf); }
pg_tensor *pg_sin(const pg_tensor *a) { return map1(a, sinf); }
pg_tensor *pg_cos(const pg_tensor *a) { return map1(a, cosf); }
pg_tensor *pg_erf(const pg_tensor *a) { return map1(a, erff); }

static float fneg(float x) { return -x; }
static float fsqrt(float x) { return sqrtf(x); }

pg_tensor *pg_neg(const pg_tensor *a) { return map1(a, fneg); }
pg_tensor *pg_abs(const pg_tensor *a) { return map1(a, fabsf); }
pg_tensor *pg_sqrt(const pg_tensor *a) { return map1(a, fsqrt); }

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
