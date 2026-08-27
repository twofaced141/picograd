#include "autograd.h"

#include "../backend/backend.h"
#include "../backend/cpu/gemm.h"
#include "../ops/activations.h"
#include "../ops/elementwise.h"
#include "../ops/matmul.h"
#include "../ops/norm.h"
#include "../ops/reduce.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static pg_node *node_wrap(pg_tensor *t, bool requires_grad)
{
    pg_node *n = calloc(1, sizeof(*n));
    if (!n) {
        pg_tensor_free(t);
        return NULL;
    }
    n->value = t;
    n->requires_grad = requires_grad;
    n->refs = 1;
    return n;
}

static pg_node *attach(pg_tensor *value, size_t nparents, pg_node **parents,
                       void (*backward)(pg_node *), void *ctx)
{
    pg_node *n = calloc(1, sizeof(*n));
    if (!n) {
        pg_tensor_free(value);
        free(ctx);
        return NULL;
    }
    n->value = value;
    n->backward = backward;
    n->ctx = ctx;
    n->refs = 1;

    if (nparents) {
        n->parents = malloc(nparents * sizeof(*n->parents));
        if (!n->parents) {
            free(n);
            pg_tensor_free(value);
            free(ctx);
            return NULL;
        }
        for (size_t i = 0; i < nparents; i++) {
            n->parents[i] = parents[i];
            pg_node_retain(parents[i]);
            n->requires_grad |= parents[i]->requires_grad;
        }
        n->nparents = nparents;
    }
    return n;
}

static pg_node *var_leaf(size_t ndim, const size_t *shape, bool requires_grad,
                         pg_tensor *(*make)(size_t, const size_t *))
{
    pg_tensor *t = make(ndim, shape);
    if (!t)
        return NULL;
    return node_wrap(t, requires_grad);
}

pg_node *pg_var_from_data(size_t ndim, const size_t *shape, const float *data,
                          bool requires_grad)
{
    pg_tensor *t = pg_tensor_from_data(ndim, shape, data);
    if (!t)
        return NULL;
    return node_wrap(t, requires_grad);
}

pg_node *pg_var_from_tensor(const pg_tensor *t, bool requires_grad)
{
    if (!t)
        return NULL;
    return pg_var_from_data(t->ndim, t->shape, t->data, requires_grad);
}

pg_node *pg_var_zeros(size_t ndim, const size_t *shape, bool requires_grad)
{
    return var_leaf(ndim, shape, requires_grad, pg_tensor_zeros);
}

pg_node *pg_var_ones(size_t ndim, const size_t *shape, bool requires_grad)
{
    return var_leaf(ndim, shape, requires_grad, pg_tensor_ones);
}

pg_node *pg_var_full(size_t ndim, const size_t *shape, float value,
                     bool requires_grad)
{
    pg_tensor *t = pg_tensor_full(ndim, shape, value);
    if (!t)
        return NULL;
    return node_wrap(t, requires_grad);
}

pg_node *pg_var_uniform(size_t ndim, const size_t *shape, float low, float high,
                        bool requires_grad)
{
    pg_tensor *t = pg_tensor_uniform(ndim, shape, low, high);
    if (!t)
        return NULL;
    return node_wrap(t, requires_grad);
}

pg_node *pg_var_normal(size_t ndim, const size_t *shape, float mean,
                       float stddev, bool requires_grad)
{
    pg_tensor *t = pg_tensor_normal(ndim, shape, mean, stddev);
    if (!t)
        return NULL;
    return node_wrap(t, requires_grad);
}

pg_node *pg_var_scalar(float value, bool requires_grad)
{
    return pg_var_full(1, (size_t[]){1}, value, requires_grad);
}

pg_node *pg_node_retain(pg_node *n)
{
    if (n)
        n->refs++;
    return n;
}

void pg_node_free(pg_node *n)
{
    if (!n)
        return;

    pg_node **stack = NULL;
    size_t sp = 0, cap = 0;

    if (cap == 0) {
        cap = 16;
        stack = malloc(cap * sizeof(*stack));
        assert(stack);
    }
    stack[sp++] = n;

    while (sp) {
        pg_node *t = stack[--sp];
        if (--t->refs)
            continue;
        pg_tensor_free(t->value);
        pg_tensor_free(t->grad);
        free(t->ctx);
        if (t->nparents) {
            if (sp + t->nparents > cap) {
                while (sp + t->nparents > cap)
                    cap *= 2;
                pg_node **ns = realloc(stack, cap * sizeof(*ns));
                assert(ns);
                stack = ns;
            }
            for (size_t i = 0; i < t->nparents; i++)
                stack[sp++] = t->parents[i];
        }
        free(t->parents);
        free(t);
    }
    free(stack);
}

static void ensure_grad(pg_node *n)
{
    assert(n && n->requires_grad);
    if (!n->grad)
        n->grad = pg_tensor_zeros(n->value->ndim, n->value->shape);
    assert(n->grad);
}

static bool try_accum_gpu(pg_tensor *dg, const pg_tensor *g, float scale)
{
    if (pg_get_device() == PG_DEV_CPU)
        return false;
    if (dg->numel > UINT_MAX || g->numel > UINT_MAX)
        return false;
    if (g->ndim > PG_MAX_OP_NDIM || dg->ndim > PG_MAX_OP_NDIM)
        return false;

    /* Build strides for accum_scatter: for broadcast (or matching shape)
     * dims where dst size is 1, the stride is 0 so all elements accumulate
     * into a single dst element; otherwise use the dst tensor stride. */
    pg_k_strides ar;
    memset(&ar, 0, sizeof(ar));
    ar.ndim = (unsigned)g->ndim;
    ar.numel = (unsigned)g->numel;

    if (pg_shape_equal(dg->ndim, dg->shape, g->ndim, g->shape)) {
        for (size_t d = 0; d < g->ndim; d++) {
            if (g->shape[d] > UINT_MAX || dg->stride[d] > UINT_MAX)
                return false;
            ar.shape[d] = (unsigned)g->shape[d];
            ar.s[d] = (unsigned)dg->stride[d];
        }
    } else {
        size_t off = g->ndim - dg->ndim;
        for (size_t d = 0; d < g->ndim; d++) {
            if (d < off) {
                ar.shape[d] = (unsigned)g->shape[d];
                continue;
            }
            size_t j = d - off;
            if (dg->shape[j] > UINT_MAX || dg->stride[j] > UINT_MAX)
                return false;
            ar.shape[d] = (unsigned)g->shape[d];
            ar.s[d] = dg->shape[j] == 1 ? 0 : (unsigned)dg->stride[j];
        }
    }

    size_t bytes_dg = dg->numel * sizeof(float);
    size_t bytes_g = g->numel * sizeof(float);
    pg_dev_buf ddg = pg_dev_buf_new(bytes_dg ? bytes_dg : 1);
    pg_dev_buf ddst_g = pg_dev_buf_new(bytes_g ? bytes_g : 1);
    if (!ddg.ptr || !ddst_g.ptr) {
        pg_dev_buf_free(&ddg);
        pg_dev_buf_free(&ddst_g);
        return false;
    }
    if (pg_copy_h2d(ddst_g.ptr, g->data, bytes_g) != PG_OK ||
        pg_copy_h2d(ddg.ptr, dg->data, bytes_dg) != PG_OK) {
        pg_dev_buf_free(&ddg);
        pg_dev_buf_free(&ddst_g);
        return false;
    }
    pg_status st = pg_op_accum_scatter(ddg.ptr, ddst_g.ptr, scale, &ar);
    if (st != PG_OK) {
        pg_dev_buf_free(&ddg);
        pg_dev_buf_free(&ddst_g);
        return false;
    }
    if (pg_dev_sync() != PG_OK) {
        pg_dev_buf_free(&ddg);
        pg_dev_buf_free(&ddst_g);
        return false;
    }
    if (pg_copy_d2h(dg->data, ddg.ptr, bytes_dg) != PG_OK) {
        pg_dev_buf_free(&ddg);
        pg_dev_buf_free(&ddst_g);
        return false;
    }
    pg_dev_buf_free(&ddg);
    pg_dev_buf_free(&ddst_g);
    return true;
}

static void accum(pg_node *dst, const pg_tensor *g, float scale)
{
    if (!dst->requires_grad || !g)
        return;

    ensure_grad(dst);
    pg_tensor *dg = dst->grad;
    assert(g->ndim >= dg->ndim);

    if (try_accum_gpu(dg, g, scale))
        return;

    if (pg_shape_equal(dg->ndim, dg->shape, g->ndim, g->shape)) {
        for (size_t i = 0; i < dg->numel; i++)
            dg->data[i] += scale * g->data[i];
        return;
    }

    size_t off = g->ndim - dg->ndim;
    size_t proj[PG_MAX_NDIM];
    for (size_t d = 0; d < g->ndim; d++) {
        if (d < off) {
            proj[d] = 0;
            continue;
        }
        size_t j = d - off;
        proj[d] = dg->shape[j] == 1 ? 0 : dg->stride[j];
    }

    size_t idx[PG_MAX_NDIM] = {0};
    size_t o = 0;
    for (size_t p = 0; p < g->numel; p++) {
        dg->data[o] += scale * g->data[p];
        for (size_t d = g->ndim; d-- > 0;) {
            idx[d]++;
            o += proj[d];
            if (idx[d] < g->shape[d])
                break;
            idx[d] = 0;
            o -= proj[d] * g->shape[d];
        }
    }
}

typedef struct {
    size_t axis;
    bool keepdim;
} red_ctx;

static pg_tensor *expand_reduce(const pg_tensor *g, const pg_tensor *like,
                                size_t axis, bool keepdim)
{
    pg_tensor *r = pg_tensor_zeros(like->ndim, like->shape);
    if (!r)
        return NULL;

    size_t sg[PG_MAX_NDIM];
    for (size_t d = 0; d < like->ndim; d++) {
        if (d == axis) {
            sg[d] = 0;
            continue;
        }
        size_t gd = keepdim ? d : (d < axis ? d : d - 1);
        sg[d] = g->stride[gd];
    }

    size_t idx[PG_MAX_NDIM] = {0};
    size_t og = 0;
    for (size_t p = 0; p < r->numel; p++) {
        r->data[p] = g->data[og];
        for (size_t d = like->ndim; d-- > 0;) {
            idx[d]++;
            og += sg[d];
            if (idx[d] < like->shape[d])
                break;
            idx[d] = 0;
            og -= sg[d] * like->shape[d];
        }
    }
    return r;
}

static void bwd_add(pg_node *n)
{
    accum(n->parents[0], n->grad, 1.0f);
    accum(n->parents[1], n->grad, 1.0f);
}

static void bwd_sub(pg_node *n)
{
    accum(n->parents[0], n->grad, 1.0f);
    accum(n->parents[1], n->grad, -1.0f);
}

static void bwd_mul(pg_node *n)
{
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    if (pa->requires_grad) {
        pg_tensor *t = pg_mul(n->grad, pb->value);
        assert(t);
        accum(pa, t, 1.0f);
        pg_tensor_free(t);
    }
    if (pb->requires_grad) {
        pg_tensor *t = pg_mul(n->grad, pa->value);
        assert(t);
        accum(pb, t, 1.0f);
        pg_tensor_free(t);
    }
}

static void bwd_div(pg_node *n)
{
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    if (pa->requires_grad) {
        pg_tensor *t = pg_div(n->grad, pb->value);
        assert(t);
        accum(pa, t, 1.0f);
        pg_tensor_free(t);
    }
    if (pb->requires_grad) {
        pg_tensor *d = pg_mul(pb->value, pb->value);
        pg_tensor *num = pg_mul(n->grad, pa->value);
        pg_tensor *t = num ? pg_div(num, d) : NULL;
        assert(t);
        accum(pb, t, -1.0f);
        pg_tensor_free(d);
        pg_tensor_free(num);
        pg_tensor_free(t);
    }
}

static void bwd_neg(pg_node *n)
{
    accum(n->parents[0], n->grad, -1.0f);
}

/* Generic backward for elementwise unary ops whose local gradient depends
 * only on the op input value x = p->value and incoming grad g = n->grad. */
typedef float (*pg_unary_local)(float x, size_t idx);

static void bwd_unary_elemwise(pg_node *n, pg_unary_local local)
{
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    ensure_grad(p);
    pg_tensor *g = n->grad, *v = p->value, *pg = p->grad;
    assert(g->numel == v->numel && pg->numel == g->numel);
    for (size_t i = 0; i < g->numel; i++)
        pg->data[i] += g->data[i] * local(v->data[i], i);
}

static float grad_exp(float x, size_t idx)      { (void)idx; return expf(x); }
static float grad_log(float x, size_t idx)      { (void)idx; return 1.0f / x; }
static float grad_sqrt(float x, size_t idx)     { (void)idx; return 0.5f / sqrtf(x); }
static float grad_sin(float x, size_t idx)      { (void)idx; return cosf(x); }
static float grad_cos(float x, size_t idx)      { (void)idx; return -sinf(x); }
static float grad_sigmoid(float x, size_t idx)   { (void)idx; { float s = 1.0f / (1.0f + expf(-x)); return s * (1.0f - s); } }
static float grad_tanh(float x, size_t idx)     { (void)idx; { float t = tanhf(x); return 1.0f - t * t; } }
static float grad_relu_local(float x, size_t idx) { (void)idx; return x > 0.0f ? 1.0f : 0.0f; }

static void bwd_exp(pg_node *n)  { bwd_unary_elemwise(n, grad_exp); }
static void bwd_log(pg_node *n)   { bwd_unary_elemwise(n, grad_log); }
static void bwd_sqrt(pg_node *n)  { bwd_unary_elemwise(n, grad_sqrt); }
static void bwd_sin(pg_node *n)   { bwd_unary_elemwise(n, grad_sin); }
static void bwd_cos(pg_node *n)   { bwd_unary_elemwise(n, grad_cos); }
static void bwd_relu(pg_node *n)  { bwd_unary_elemwise(n, grad_relu_local); }
static void bwd_sigmoid(pg_node *n) { bwd_unary_elemwise(n, grad_sigmoid); }
static void bwd_tanh(pg_node *n)  { bwd_unary_elemwise(n, grad_tanh); }

static void bwd_matmul(pg_node *n)
{
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    const pg_tensor *a = pa->value, *b = pb->value, *g = n->grad;

    bool av = a->ndim == 1, bv = b->ndim == 1;
    size_t am = av ? 1 : a->shape[a->ndim - 2];
    size_t ak = a->shape[a->ndim - 1];
    size_t bn = bv ? 1 : b->shape[b->ndim - 1];

    bool abat = !av && a->ndim > 2;
    bool bbat = !bv && b->ndim > 2;

    size_t nbatch = 1;
    if (abat)
        for (size_t d = 0; d < a->ndim - 2; d++)
            nbatch *= a->shape[d];
    else if (bbat)
        for (size_t d = 0; d < b->ndim - 2; d++)
            nbatch *= b->shape[d];

    size_t step_a = abat ? am * ak : 0;
    size_t step_b = bbat ? ak * bn : 0;
    size_t step_g = am * bn;

    if (!pa->requires_grad && !pb->requires_grad)
        return;
    if (av && bv) {
        if (pa->requires_grad) {
            ensure_grad(pa);
            for (size_t i = 0; i < ak; i++)
                pa->grad->data[i] += g->data[0] * b->data[i];
        }
        if (pb->requires_grad) {
            ensure_grad(pb);
            for (size_t i = 0; i < ak; i++)
                pb->grad->data[i] += g->data[0] * a->data[i];
        }
        return;
    }

    pg_tensor *ga = pa->requires_grad ? (ensure_grad(pa), pa->grad) : NULL;
    pg_tensor *gb = pb->requires_grad ? (ensure_grad(pb), pb->grad) : NULL;

    // Use GEMM for large problems, fallback to naive for tiny where temp alloc overhead > compute
    bool use_gemm = (am * bn * ak > 4096) && !av && !bv;
    float *tmp_trans = NULL;
    float *tmp_out = NULL;
    size_t tmp_trans_cap = 0, tmp_out_cap = 0;
    if (use_gemm) {
        size_t need_trans = ak * bn > am * ak ? ak * bn : am * ak;
        size_t need_out = am * ak > ak * bn ? am * ak : ak * bn;
        tmp_trans = malloc(need_trans * sizeof(float));
        tmp_out = malloc(need_out * sizeof(float));
        if (!tmp_trans || !tmp_out) {
            free(tmp_trans); free(tmp_out);
            tmp_trans = NULL; tmp_out = NULL;
            use_gemm = false;
        } else {
            tmp_trans_cap = need_trans;
            tmp_out_cap = need_out;
        }
    }

    for (size_t s = 0; s < nbatch; s++) {
        const float *ap = a->data + s * step_a;
        const float *bp = b->data + s * step_b;
        const float *gp = g->data + s * step_g;
        float *gap = ga ? ga->data + (av ? 0 : s * step_a) : NULL;
        float *gbp = gb ? gb->data + (bv ? 0 : s * step_b) : NULL;

        if (gap) {
            if (use_gemm) {
                // gap += grad * B^T   grad[am,bn] * B^T[bn,ak]
                // transpose B [ak,bn] -> [bn,ak]
                // tmp_trans = B^T
                for (size_t i = 0; i < ak; i++) {
                    for (size_t j = 0; j < bn; j++) tmp_trans[j * ak + i] = bp[i * bn + j];
                }
                // tmp_out = grad * B^T
                // zero tmp_out
                for (size_t i = 0; i < am * ak; i++) tmp_out[i] = 0.0f;
                pg_cpu_gemm(am, ak, bn, gp, bn, tmp_trans, ak, tmp_out, ak);
                #pragma GCC ivdep
                for (size_t i = 0; i < am * ak; i++) gap[i] += tmp_out[i];
            } else {
                for (size_t i = 0; i < am; i++)
                    for (size_t p2 = 0; p2 < ak; p2++) {
                        float acc = 0.0f;
                        for (size_t j = 0; j < bn; j++)
                            acc += gp[i * bn + j] * bp[p2 * bn + j];
                        gap[i * ak + p2] += acc;
                    }
            }
        }

        if (gbp) {
            if (use_gemm) {
                // gbp += A^T * grad   A^T[ak,am] * grad[am,bn]
                for (size_t i = 0; i < am; i++) {
                    for (size_t j = 0; j < ak; j++) tmp_trans[j * am + i] = ap[i * ak + j];
                }
                for (size_t i = 0; i < ak * bn; i++) tmp_out[i] = 0.0f;
                pg_cpu_gemm(ak, bn, am, tmp_trans, am, gp, bn, tmp_out, bn);
                #pragma GCC ivdep
                for (size_t i = 0; i < ak * bn; i++) gbp[i] += tmp_out[i];
            } else {
                for (size_t p2 = 0; p2 < ak; p2++)
                    for (size_t j = 0; j < bn; j++) {
                        float acc = 0.0f;
                        for (size_t i = 0; i < am; i++)
                            acc += ap[i * ak + p2] * gp[i * bn + j];
                        gbp[p2 * bn + j] += acc;
                    }
            }
        }
    }
    free(tmp_trans);
    free(tmp_out);
}

static void bwd_reduce(pg_node *n, float scale)
{
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    red_ctx *cx = n->ctx;
    pg_tensor *e = expand_reduce(n->grad, p->value, cx->axis, cx->keepdim);
    assert(e);
    accum(p, e, scale);
    pg_tensor_free(e);
}

static void bwd_sum(pg_node *n)
{
    bwd_reduce(n, 1.0f);
}

static void bwd_mean(pg_node *n)
{
    pg_node *p = n->parents[0];
    red_ctx *cx = n->ctx;
    bwd_reduce(n, 1.0f / (float)p->value->shape[cx->axis]);
}

static void bwd_softmax(pg_node *n)
{
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    red_ctx *cx = n->ctx;

    pg_tensor *go = pg_mul(n->grad, n->value);
    pg_tensor *s = go ? pg_sum(go, cx->axis, true) : NULL;
    pg_tensor *se = s ? expand_reduce(s, n->value, cx->axis, true) : NULL;
    pg_tensor *diff = se ? pg_sub(n->grad, se) : NULL;
    pg_tensor *gx = diff ? pg_mul(n->value, diff) : NULL;
    assert(gx);

    accum(p, gx, 1.0f);
    pg_tensor_free(go);
    pg_tensor_free(s);
    pg_tensor_free(se);
    pg_tensor_free(diff);
    pg_tensor_free(gx);
}

#define BINARY_OP(agn, opn, bwd)                                   \
    pg_node *agn(pg_node *a, pg_node *b)                           \
    {                                                              \
        assert(a && b && a->value && b->value);                    \
        pg_tensor *v = opn(a->value, b->value);                    \
        if (!v)                                                    \
            return NULL;                                           \
        return attach(v, 2, (pg_node *[]){a, b}, bwd, NULL);       \
    }

BINARY_OP(pg_ag_add, pg_add, bwd_add)
BINARY_OP(pg_ag_sub, pg_sub, bwd_sub)
BINARY_OP(pg_ag_mul, pg_mul, bwd_mul)
BINARY_OP(pg_ag_div, pg_div, bwd_div)

#define UNARY_OP(agn, opn, bwd)                                    \
    pg_node *agn(pg_node *a)                                       \
    {                                                              \
        assert(a && a->value);                                     \
        pg_tensor *v = opn(a->value);                              \
        if (!v)                                                    \
            return NULL;                                           \
        return attach(v, 1, &a, bwd, NULL);                        \
    }

UNARY_OP(pg_ag_neg, pg_neg, bwd_neg)
UNARY_OP(pg_ag_exp, pg_exp, bwd_exp)
UNARY_OP(pg_ag_log, pg_log, bwd_log)
UNARY_OP(pg_ag_sqrt, pg_sqrt, bwd_sqrt)
UNARY_OP(pg_ag_sin, pg_sin, bwd_sin)
UNARY_OP(pg_ag_cos, pg_cos, bwd_cos)
UNARY_OP(pg_ag_relu, pg_relu, bwd_relu)
UNARY_OP(pg_ag_sigmoid, pg_sigmoid, bwd_sigmoid)
UNARY_OP(pg_ag_tanh, pg_tanh, bwd_tanh)

pg_node *pg_ag_matmul(pg_node *a, pg_node *b)
{
    assert(a && b && a->value && b->value);
    pg_tensor *v = pg_matmul(a->value, b->value);
    if (!v)
        return NULL;
    return attach(v, 2, (pg_node *[]){a, b}, bwd_matmul, NULL);
}

static pg_node *ag_reduce(pg_node *a, size_t axis, bool keepdim, bool mean)
{
    assert(a && a->value && a->value->ndim >= 1 && axis < a->value->ndim);

    red_ctx *cx = malloc(sizeof(*cx));
    if (!cx)
        return NULL;
    cx->axis = axis;
    cx->keepdim = keepdim;

    pg_tensor *v = mean ? pg_mean(a->value, axis, keepdim)
                        : pg_sum(a->value, axis, keepdim);
    if (!v) {
        free(cx);
        return NULL;
    }

    pg_node *r = attach(v, 1, &a, mean ? bwd_mean : bwd_sum, cx);
    if (!r)
        free(cx);
    return r;
}

pg_node *pg_ag_sum(pg_node *a, size_t axis, bool keepdim)
{
    return ag_reduce(a, axis, keepdim, false);
}

pg_node *pg_ag_mean(pg_node *a, size_t axis, bool keepdim)
{
    return ag_reduce(a, axis, keepdim, true);
}

static pg_node *reduce_all(pg_node *a, bool mean)
{
    assert(a && a->value && a->value->ndim >= 1);

    pg_node *cur = pg_node_retain(a);
    for (size_t d = a->value->ndim; d > 1; d--) {
        pg_node *next = pg_ag_sum(cur, 0, false);
        pg_node_free(cur);
        if (!next)
            return NULL;
        cur = next;
    }
    pg_node *out = mean ? pg_ag_mean(cur, 0, false) : pg_ag_sum(cur, 0, false);
    pg_node_free(cur);
    return out;
}

pg_node *pg_ag_sum_all(pg_node *a)
{
    return reduce_all(a, false);
}

pg_node *pg_ag_mean_all(pg_node *a)
{
    return reduce_all(a, true);
}

pg_node *pg_ag_softmax(pg_node *a, size_t axis)
{
    assert(a && a->value && axis < a->value->ndim);

    red_ctx *cx = malloc(sizeof(*cx));
    if (!cx)
        return NULL;
    cx->axis = axis;
    cx->keepdim = true;

    pg_tensor *v = pg_softmax(a->value, axis);
    if (!v) {
        free(cx);
        return NULL;
    }

    pg_node *r = attach(v, 1, &a, bwd_softmax, cx);
    if (!r)
        free(cx);
    return r;
}

// ---------- layernorm / rmsnorm ----------
typedef struct { float eps; size_t N; bool has_w; bool has_b; } ln_ctx_t;
typedef struct { float eps; size_t N; bool has_w; } rms_ctx_t;

static void bwd_layernorm(pg_node *n){
    ln_ctx_t *cx = (ln_ctx_t*)n->ctx;
    pg_node *px = n->parents[0];
    pg_node *pw = NULL, *pb = NULL;
    size_t idx=1;
    if(cx->has_w){ pw = n->parents[idx++]; }
    if(cx->has_b){ pb = n->parents[idx++]; }
    size_t N = cx->N;
    float eps = cx->eps;
    const pg_tensor *x = px->value;
    const pg_tensor *g = n->grad; // grad_y
    size_t outer = x->numel / N;
    const float *w = NULL;
    if(pw && pw->value) w = pw->value->data;
    // grad for weight/bias
    if(pw && pw->requires_grad){
        ensure_grad(pw);
        // need xn = (x - mean)*rstd
        for(size_t o=0;o<outer;o++){
            const float *row = x->data + o*N;
            const float *grow = g->data + o*N;
            double sum=0; for(size_t i=0;i<N;i++) sum+=row[i];
            float mean=(float)(sum/N);
            double var=0; for(size_t i=0;i<N;i++){ double d=row[i]-mean; var+=d*d; } var/=N;
            float rstd=1.0f/sqrtf((float)var + eps);
            for(size_t i=0;i<N;i++){
                float xn=(row[i]-mean)*rstd;
                pw->grad->data[i] += grow[i] * xn;
            }
        }
    }
    if(pb && pb->requires_grad){
        ensure_grad(pb);
        for(size_t o=0;o<outer;o++){
            const float *grow=g->data + o*N;
            for(size_t i=0;i<N;i++) pb->grad->data[i] += grow[i];
        }
    }
    if(!px->requires_grad) return;
    ensure_grad(px);
    // grad_x as per formula
    for(size_t o=0;o<outer;o++){
        const float *row=x->data + o*N;
        const float *grow=g->data + o*N;
        float *gxrow=px->grad->data + o*N;
        double sum=0; for(size_t i=0;i<N;i++) sum+=row[i];
        float mean=(float)(sum/N);
        double var=0; for(size_t i=0;i<N;i++){ double d=row[i]-mean; var+=d*d; } var/=N;
        float rstd=1.0f/sqrtf((float)var + eps);
        float invN = 1.0f / (float)N;
        // compute xn
        // we can compute on fly
        // need sums: sum_gw and sum_gw_xn
        float sum_gw=0, sum_gw_xn=0;
        for(size_t i=0;i<N;i++){
            float gw = grow[i] * (w? w[i]:1.0f);
            sum_gw += gw;
            float xn=(row[i]-mean)*rstd;
            sum_gw_xn += gw * xn;
        }
        for(size_t i=0;i<N;i++){
            float xn=(row[i]-mean)*rstd;
            float gw = grow[i] * (w? w[i]:1.0f);
            float dx = rstd * (gw - invN*sum_gw - xn*invN*sum_gw_xn);
            gxrow[i] += dx;
        }
    }
}

static void bwd_rmsnorm(pg_node *n){
    rms_ctx_t *cx=(rms_ctx_t*)n->ctx;
    size_t N=cx->N; float eps=cx->eps;
    pg_node *px=n->parents[0];
    pg_node *pw=cx->has_w && n->nparents>1? n->parents[1]:NULL;
    const pg_tensor *x=px->value;
    const pg_tensor *g=n->grad;
    size_t outer=x->numel / N;
    const float *w = (pw && pw->value)? pw->value->data:NULL;
    if(pw && pw->requires_grad){
        ensure_grad(pw);
        for(size_t o=0;o<outer;o++){
            const float *row=x->data+o*N;
            const float *grow=g->data+o*N;
            double sumsq=0; for(size_t i=0;i<N;i++) sumsq+=(double)row[i]*row[i];
            float rstd=1.0f/sqrtf((float)(sumsq/N)+eps);
            for(size_t i=0;i<N;i++) pw->grad->data[i] += grow[i] * row[i]*rstd;
        }
    }
    if(!px->requires_grad) return;
    ensure_grad(px);
    for(size_t o=0;o<outer;o++){
        const float *row=x->data+o*N;
        const float *grow=g->data+o*N;
        float *gxrow=px->grad->data+o*N;
        double sumsq=0; for(size_t i=0;i<N;i++) sumsq+=(double)row[i]*row[i];
        float var=(float)(sumsq/N);
        float rstd=1.0f/sqrtf(var+eps);
        float invN = 1.0f/(float)N;
        // need sum_gw_x for rmsnorm grad
        float sum_gw_x=0;
        for(size_t i=0;i<N;i++){
            float gw=grow[i]*(w?w[i]:1.0f);
            sum_gw_x += gw * row[i];
        }
        float scale = rstd;
        float coeff = - sum_gw_x * rstd * rstd * rstd * invN; // derivative of rstd w.r.t x includes sum term
        // simplified formula: dx = rstd*gw - x * (rstd^3 * invN * sum_gw_x)
        // where gw = grad_y * w
        for(size_t i=0;i<N;i++){
            float gw=grow[i]*(w?w[i]:1.0f);
            float dx = gw * rstd + row[i] * coeff;
            gxrow[i] += dx;
        }
    }
}

pg_node *pg_ag_layernorm(pg_node *x, pg_node *weight, pg_node *bias, float eps){
    assert(x && x->value);
    const pg_tensor *w = weight ? weight->value : NULL;
    const pg_tensor *b = bias ? bias->value : NULL;
    pg_tensor *v = pg_layernorm(x->value, w, b, eps);
    if(!v) return NULL;
    ln_ctx_t *cx=malloc(sizeof(*cx));
    if(!cx){ pg_tensor_free(v); return NULL; }
    cx->eps=eps;
    cx->N=x->value->shape[x->value->ndim-1];
    cx->has_w = weight != NULL;
    cx->has_b = bias != NULL;
    size_t npar = 1 + (weight?1:0) + (bias?1:0);
    pg_node *pars[3]; size_t idx=0; pars[idx++]=x; if(weight) pars[idx++]=weight; if(bias) pars[idx++]=bias;
    pg_node *r=attach(v, npar, pars, bwd_layernorm, cx);
    if(!r) free(cx);
    return r;
}
pg_node *pg_ag_rmsnorm(pg_node *x, pg_node *weight, float eps){
    assert(x && x->value);
    const pg_tensor *w = weight ? weight->value : NULL;
    pg_tensor *v = pg_rmsnorm(x->value, w, eps);
    if(!v) return NULL;
    rms_ctx_t *cx=malloc(sizeof(*cx));
    if(!cx){ pg_tensor_free(v); return NULL; }
    cx->eps=eps; cx->N=x->value->shape[x->value->ndim-1];
    cx->has_w = weight != NULL;
    size_t npar = 1 + (weight?1:0);
    pg_node *pars[2]; pars[0]=x; if(weight) pars[1]=weight;
    pg_node *r=attach(v, npar, pars, bwd_rmsnorm, cx);
    if(!r) free(cx);
    return r;
}

typedef struct {
    pg_node **items;
    size_t n, cap;
} node_vec;

static void vec_push(node_vec *v, pg_node *n)
{
    if (v->n == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 32;
        pg_node **ni = realloc(v->items, nc * sizeof(*ni));
        assert(ni);
        v->items = ni;
        v->cap = nc;
    }
    v->items[v->n++] = n;
}

static void topo_sort(pg_node *root, node_vec *order)
{
    typedef struct {
        pg_node *n;
        size_t i;
    } frame;

    frame *stk = NULL;
    size_t sp = 0, cap = 0;

    root->mark = 1;
    do {
        if (sp == cap) {
            cap = cap ? cap * 2 : 32;
            frame *ns = realloc(stk, cap * sizeof(*ns));
            assert(ns);
            stk = ns;
        }
        stk[sp].n = root;
        stk[sp].i = 0;
        sp++;
    } while (0);

    while (sp) {
        frame *f = &stk[sp - 1];
        if (f->i < f->n->nparents) {
            pg_node *c = f->n->parents[f->i++];
            if (c->requires_grad && !c->mark) {
                c->mark = 1;
                if (sp == cap) {
                    cap = cap ? cap * 2 : 32;
                    frame *ns = realloc(stk, cap * sizeof(*ns));
                    assert(ns);
                    stk = ns;
                }
                stk[sp].n = c;
                stk[sp].i = 0;
                sp++;
            }
        } else {
            vec_push(order, f->n);
            sp--;
        }
    }
    free(stk);
}

void pg_backward(pg_node *loss)
{
    if (!loss || !loss->requires_grad)
        return;

    node_vec order = {0};
    topo_sort(loss, &order);

    for (size_t i = 0; i < order.n; i++)
        if (order.items[i]->grad)
            pg_tensor_fill(order.items[i]->grad, 0.0f);

    ensure_grad(loss);
    pg_tensor_fill(loss->grad, 1.0f);

    for (size_t i = order.n; i-- > 0;) {
        pg_node *n = order.items[i];
        if (n->backward && n->grad)
            n->backward(n);
    }

    for (size_t i = 0; i < order.n; i++)
        order.items[i]->mark = 0;

    free(order.items);
}
