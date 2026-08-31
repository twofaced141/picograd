#include "autograd.h"

#include "../backend/backend.h"
#include "../backend/cpu/gemm.h"
#include "../core/convert.h"
#include "../ops/activations.h"
#include "../ops/conv.h"
#include "../ops/elementwise.h"
#include "../ops/index.h"
#include "../ops/matmul.h"
#include "../ops/norm.h"
#include "../ops/reduce.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../jit/jit.h"

/* ---- JIT autograd state ---- */
static bool g_ag_jit_enabled = true;
static char g_ag_last_err[512] = {0};
static bool g_ag_last_was_jit = false;
static size_t g_ag_jit_hits = 0, g_ag_jit_fallbacks = 0;

/* ---- grad enabled / no_grad ---- */
static bool g_grad_enabled = true;
static int g_no_grad_depth = 0;

bool pg_autograd_is_grad_enabled(void){ return g_grad_enabled && g_no_grad_depth==0; }
void pg_autograd_set_grad_enabled(bool enabled){ g_grad_enabled = enabled; }
void pg_no_grad_push(void){ g_no_grad_depth++; }
void pg_no_grad_pop(void){ if(g_no_grad_depth>0) g_no_grad_depth--; }

static void ag_set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_ag_last_err, sizeof(g_ag_last_err), fmt, ap);
    va_end(ap);
}

static bool ag_op_is_jit_elementwise(pg_ag_op_t op) {
    switch (op) {
        case PG_AG_OP_ADD: case PG_AG_OP_SUB: case PG_AG_OP_MUL: case PG_AG_OP_DIV:
        case PG_AG_OP_NEG: case PG_AG_OP_EXP: case PG_AG_OP_LOG: case PG_AG_OP_SQRT:
        case PG_AG_OP_SIN: case PG_AG_OP_COS: case PG_AG_OP_RELU:
        case PG_AG_OP_SIGMOID: case PG_AG_OP_TANH:
            return true;
        default: return false;
    }
}

void pg_autograd_set_jit(bool enabled) { g_ag_jit_enabled = enabled; }
bool pg_autograd_is_jit_enabled(void) { return g_ag_jit_enabled; }
const char *pg_autograd_last_error(void) { return g_ag_last_err; }
bool pg_autograd_last_was_jit(void) { return g_ag_last_was_jit; }
size_t pg_autograd_jit_hits(void) { return g_ag_jit_hits; }
size_t pg_autograd_jit_fallbacks(void) { return g_ag_jit_fallbacks; }

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
    n->ag_op = PG_AG_OP_NONE;
    return n;
}

static pg_node *attach(pg_tensor *value, size_t nparents, pg_node **parents,
                       void (*backward)(pg_node *), void *ctx, pg_ag_op_t ag_op)
{
    // no_grad mode: create detached node, no graph
    if(!pg_autograd_is_grad_enabled()){
        free(ctx);
        pg_node *n = node_wrap(value, false);
        // do not retain parents, no backward
        return n;
    }
    pg_node *n = calloc(1, sizeof(*n));
    if (!n) {
        pg_tensor_free(value);
        free(ctx);
        return NULL;
    }
    n->value = value;
    n->backward = backward;
    n->ctx = ctx;
    n->ag_op = ag_op;
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
        if(!n->requires_grad){ n->backward=NULL; free(ctx); n->ctx=NULL; }
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

pg_node *pg_node_detach(pg_node *n){
    if(!n || !n->value) return NULL;
    pg_tensor *t = pg_tensor_clone(n->value);
    if(!t) return NULL;
    return node_wrap(t, false);
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
        if (!stack) {
            if (--n->refs)
                return;
            pg_tensor_free(n->value);
            pg_tensor_free(n->grad);
            free(n->ctx);
            for (size_t i = 0; i < n->nparents; i++)
                pg_node_free(n->parents[i]);
            free(n->parents);
            free(n);
            return;
        }
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
                size_t need = sp + t->nparents;
                size_t ncap = cap;
                while (ncap < need)
                    ncap *= 2;
                pg_node **ns = realloc(stack, ncap * sizeof(*ns));
                if (!ns) {
                    for (size_t i = 0; i < t->nparents; i++)
                        pg_node_free(t->parents[i]);
                    free(t->parents);
                    free(t);
                    continue;
                }
                stack = ns;
                cap = ncap;
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
    // small tensors: H2D/D2H overhead dominates, stay on CPU
    if (g->numel < 65536 && dg->numel < 65536)
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

static inline bool ag_is_contiguous(const pg_tensor *t) {
    size_t acc=1;
    for (size_t i=t->ndim; i-- >0;) {
        if (t->stride[i]!=acc) return false;
        acc*=t->shape[i];
    }
    return true;
}
static void bwd_mul(pg_node *n)
{
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    const pg_tensor *g = n->grad;
    // fast path: all same shape contiguous
    if (pa->requires_grad && pg_shape_equal(pa->value->ndim, pa->value->shape, g->ndim, g->shape) &&
        pg_shape_equal(pb->value->ndim, pb->value->shape, g->ndim, g->shape) &&
        ag_is_contiguous(pa->value) && ag_is_contiguous(pb->value) &&
        ag_is_contiguous(g) && ag_is_contiguous(pa->grad ? pa->grad : pa->value)) {
        // direct without alloc if pa grad already allocated or can be ensured
        if (pa->grad || pa->requires_grad) {
            ensure_grad(pa);
            for (size_t i=0;i<g->numel;i++) pa->grad->data[i] += g->data[i] * pb->value->data[i];
        }
    } else if (pa->requires_grad) {
        pg_tensor *t = pg_mul(g, pb->value);
        if (t) { accum(pa, t, 1.0f); pg_tensor_free(t); }
    }
    if (pb->requires_grad && pg_shape_equal(pb->value->ndim, pb->value->shape, g->ndim, g->shape) &&
        pg_shape_equal(pa->value->ndim, pa->value->shape, g->ndim, g->shape) &&
        ag_is_contiguous(pa->value) && ag_is_contiguous(pb->value) && ag_is_contiguous(g)) {
        ensure_grad(pb);
        for (size_t i=0;i<g->numel;i++) pb->grad->data[i] += g->data[i] * pa->value->data[i];
    } else if (pb->requires_grad) {
        pg_tensor *t = pg_mul(g, pa->value);
        if (t) { accum(pb, t, 1.0f); pg_tensor_free(t); }
    }
}

static void bwd_div(pg_node *n)
{
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    const pg_tensor *g = n->grad;
    if (pa->requires_grad) {
        if (pg_shape_equal(pa->value->ndim, pa->value->shape, g->ndim, g->shape) &&
            pg_shape_equal(pb->value->ndim, pb->value->shape, g->ndim, g->shape) &&
            ag_is_contiguous(g) && ag_is_contiguous(pb->value) && ag_is_contiguous(pa->value)) {
            ensure_grad(pa);
            for (size_t i=0;i<g->numel;i++) pa->grad->data[i] += g->data[i] / pb->value->data[i];
        } else {
            pg_tensor *t = pg_div(g, pb->value);
            if (t) { accum(pa, t, 1.0f); pg_tensor_free(t); }
        }
    }
    if (pb->requires_grad) {
        if (pg_shape_equal(pa->value->ndim, pa->value->shape, g->ndim, g->shape) &&
            pg_shape_equal(pb->value->ndim, pb->value->shape, g->ndim, g->shape) &&
            ag_is_contiguous(g) && ag_is_contiguous(pa->value) && ag_is_contiguous(pb->value)) {
            ensure_grad(pb);
            for (size_t i=0;i<g->numel;i++) {
                float b = pb->value->data[i];
                float grad = - g->data[i] * pa->value->data[i] / (b*b);
                if (isfinite(grad)) pb->grad->data[i] += grad;
            }
        } else {
            pg_tensor *d = pg_mul(pb->value, pb->value);
            pg_tensor *num = d ? pg_mul(g, pa->value) : NULL;
            pg_tensor *t = (num && d) ? pg_div(num, d) : NULL;
            if (t) accum(pb, t, -1.0f);
            pg_tensor_free(d);
            pg_tensor_free(num);
            if (t) pg_tensor_free(t);
        }
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
static float grad_sigmoid(float x, size_t idx) {
    (void)idx;
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f - s);
}
static float grad_tanh(float x, size_t idx) {
    (void)idx;
    float t = tanhf(x);
    return 1.0f - t * t;
}
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
    const pg_tensor *a_orig = pa->value, *b_orig = pb->value;
    const pg_tensor *g = n->grad;
    // handle mixed-precision: convert half inputs to f32 temporaries for gradient GEMM (f32 accum)
    pg_tensor *a_conv = NULL, *b_conv = NULL;
    const pg_tensor *a = a_orig, *b = b_orig;
    if (a_orig->dtype != PG_DTYPE_F32) {
        a_conv = pg_tensor_empty_dtype(PG_DTYPE_F32, a_orig->ndim, a_orig->shape);
        if (!a_conv) return;
        for (size_t i=0;i<a_orig->numel;i++) {
            float v;
            if (a_orig->dtype==PG_DTYPE_F16) v = pg_f16_to_f32_scalar(a_orig->data_u16[i]);
            else v = pg_bf16_to_f32_scalar(a_orig->data_u16[i]);
            a_conv->data[i]=v;
        }
        a = a_conv;
    }
    if (b_orig->dtype != PG_DTYPE_F32) {
        b_conv = pg_tensor_empty_dtype(PG_DTYPE_F32, b_orig->ndim, b_orig->shape);
        if (!b_conv) { pg_tensor_free(a_conv); return; }
        for (size_t i=0;i<b_orig->numel;i++) {
            float v;
            if (b_orig->dtype==PG_DTYPE_F16) v = pg_f16_to_f32_scalar(b_orig->data_u16[i]);
            else v = pg_bf16_to_f32_scalar(b_orig->data_u16[i]);
            b_conv->data[i]=v;
        }
        b = b_conv;
    }

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

    if (!pa->requires_grad && !pb->requires_grad) {
        pg_tensor_free(a_conv); pg_tensor_free(b_conv);
        return;
    }
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
        pg_tensor_free(a_conv); pg_tensor_free(b_conv);
        return;
    }

    pg_tensor *ga = pa->requires_grad ? (ensure_grad(pa), pa->grad) : NULL;
    pg_tensor *gb = pb->requires_grad ? (ensure_grad(pb), pb->grad) : NULL;

    // Use GEMM for large problems, fallback to naive for tiny where temp alloc overhead > compute
    bool use_gemm = (am * bn * ak > 4096) && !av && !bv;
    float *tmp_trans = NULL;
    float *tmp_out = NULL;
    if (use_gemm) {
        size_t need_trans = ak * bn > am * ak ? ak * bn : am * ak;
        size_t need_out = am * ak > ak * bn ? am * ak : ak * bn;
        tmp_trans = malloc(need_trans * sizeof(float));
        tmp_out = malloc(need_out * sizeof(float));
        if (!tmp_trans || !tmp_out) {
            free(tmp_trans); free(tmp_out);
            tmp_trans = NULL; tmp_out = NULL;
            use_gemm = false;
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
    pg_tensor_free(a_conv); pg_tensor_free(b_conv);
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

#define BINARY_OP(agn, opn, bwd, op_enum)                             \
    pg_node *agn(pg_node *a, pg_node *b)                           \
    {                                                              \
        assert(a && b && a->value && b->value);                    \
        pg_tensor *v = opn(a->value, b->value);                    \
        if (!v)                                                    \
            return NULL;                                           \
        return attach(v, 2, (pg_node *[]){a, b}, bwd, NULL, op_enum); \
    }

BINARY_OP(pg_ag_add, pg_add, bwd_add, PG_AG_OP_ADD)
BINARY_OP(pg_ag_sub, pg_sub, bwd_sub, PG_AG_OP_SUB)
BINARY_OP(pg_ag_mul, pg_mul, bwd_mul, PG_AG_OP_MUL)
BINARY_OP(pg_ag_div, pg_div, bwd_div, PG_AG_OP_DIV)

#define UNARY_OP(agn, opn, bwd, op_enum)                               \
    pg_node *agn(pg_node *a)                                       \
    {                                                              \
        assert(a && a->value);                                     \
        pg_tensor *v = opn(a->value);                              \
        if (!v)                                                    \
            return NULL;                                           \
        return attach(v, 1, &a, bwd, NULL, op_enum);               \
    }

UNARY_OP(pg_ag_neg, pg_neg, bwd_neg, PG_AG_OP_NEG)
UNARY_OP(pg_ag_exp, pg_exp, bwd_exp, PG_AG_OP_EXP)
UNARY_OP(pg_ag_log, pg_log, bwd_log, PG_AG_OP_LOG)
UNARY_OP(pg_ag_sqrt, pg_sqrt, bwd_sqrt, PG_AG_OP_SQRT)
UNARY_OP(pg_ag_sin, pg_sin, bwd_sin, PG_AG_OP_SIN)
UNARY_OP(pg_ag_cos, pg_cos, bwd_cos, PG_AG_OP_COS)
UNARY_OP(pg_ag_relu, pg_relu, bwd_relu, PG_AG_OP_RELU)
UNARY_OP(pg_ag_sigmoid, pg_sigmoid, bwd_sigmoid, PG_AG_OP_SIGMOID)
UNARY_OP(pg_ag_tanh, pg_tanh, bwd_tanh, PG_AG_OP_TANH)

static float grad_abs(float x, size_t idx) {
    (void)idx;
    return x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f);
}
static float grad_erf(float x, size_t idx) { (void)idx; return 1.12837916709551f * expf(-x*x); }
static float grad_gelu(float x, size_t idx) {
    (void)idx;
    const float sqrt_2pi = 2.50662827463f;
    const float inv_sqrt2 = 0.70710678118f;
    float erf_term = erff(x * inv_sqrt2);
    float phi = expf(-0.5f * x * x) / sqrt_2pi;
    return 0.5f * (1.0f + erf_term) + x * phi;
}
static void bwd_abs(pg_node *n)  { bwd_unary_elemwise(n, grad_abs); }
static void bwd_erf(pg_node *n)  { bwd_unary_elemwise(n, grad_erf); }
static void bwd_gelu(pg_node *n) { bwd_unary_elemwise(n, grad_gelu); }

typedef struct { float lo; float hi; } clamp_ctx_t;
static void bwd_clamp(pg_node *n) {
    clamp_ctx_t *cx = (clamp_ctx_t*)n->ctx;
    pg_node *p = n->parents[0];
    if (!p->requires_grad) return;
    ensure_grad(p);
    pg_tensor *g = n->grad, *v = p->value, *pg = p->grad;
    for (size_t i=0;i<g->numel;i++) {
        float x = v->data[i];
        if (x >= cx->lo && x <= cx->hi) pg->data[i] += g->data[i];
    }
}
typedef struct { float alpha; } leaky_ctx_t;
static void bwd_leaky(pg_node *n) {
    leaky_ctx_t *cx = (leaky_ctx_t*)n->ctx;
    pg_node *p = n->parents[0];
    if (!p->requires_grad) return;
    ensure_grad(p);
    pg_tensor *g = n->grad, *v = p->value, *pg = p->grad;
    for (size_t i=0;i<g->numel;i++) {
        float x = v->data[i];
        pg->data[i] += g->data[i] * (x > 0 ? 1.0f : cx->alpha);
    }
}
static void bwd_pow(pg_node *n) {
    pg_node *pa = n->parents[0], *pb = n->parents[1];
    const pg_tensor *a = pa->value, *b = pb->value;
    const pg_tensor *g = n->grad;
    if (pa->requires_grad) {
        pg_tensor *one = pg_tensor_full(1, (size_t[]){1}, 1.0f);
        pg_tensor *b_minus1 = one ? pg_sub(b, one) : NULL;
        if (one) pg_tensor_free(one);
        pg_tensor *a_pow = b_minus1 ? pg_pow(a, b_minus1) : NULL;
        if (b_minus1) pg_tensor_free(b_minus1);
        if (a_pow) {
            pg_tensor *tmp1 = pg_mul(b, a_pow);
            if (tmp1) {
                pg_tensor *grad_a = pg_mul(g, tmp1);
                if (grad_a) { accum(pa, grad_a, 1.0f); pg_tensor_free(grad_a); }
                pg_tensor_free(tmp1);
            }
            pg_tensor_free(a_pow);
        }
    }
    if (pb->requires_grad) {
        // grad_b = g * a^b * log(a)
        pg_tensor *log_a = pg_log(a);
        if (!log_a) return;
        pg_tensor *pow_val = pg_pow(a, b);
        if (!pow_val) { pg_tensor_free(log_a); return; }
        pg_tensor *tmp = pg_mul(pow_val, log_a);
        pg_tensor *grad_b_un = NULL;
        if (tmp) grad_b_un = pg_mul(g, tmp);
        if (grad_b_un) { accum(pb, grad_b_un, 1.0f); pg_tensor_free(grad_b_un); }
        pg_tensor_free(tmp);
        pg_tensor_free(pow_val);
        pg_tensor_free(log_a);
    }
}

pg_node *pg_ag_pow(pg_node *a, pg_node *b) {
    if (!a || !b || !a->value || !b->value) return NULL;
    pg_tensor *v = pg_pow(a->value, b->value);
    if (!v) return NULL;
    return attach(v, 2, (pg_node*[]){a,b}, bwd_pow, NULL, PG_AG_OP_POW);
}
UNARY_OP(pg_ag_abs, pg_abs, bwd_abs, PG_AG_OP_ABS)
UNARY_OP(pg_ag_erf, pg_erf, bwd_erf, PG_AG_OP_ERF)
UNARY_OP(pg_ag_gelu, pg_gelu, bwd_gelu, PG_AG_OP_GELU)
pg_node *pg_ag_clamp(pg_node *a, float lo, float hi) {
    if (!a || !a->value) return NULL;
    if (lo > hi) return NULL;
    pg_tensor *v = pg_clamp(a->value, lo, hi);
    if (!v) return NULL;
    clamp_ctx_t *cx = malloc(sizeof(*cx));
    if (!cx) { pg_tensor_free(v); return NULL; }
    cx->lo = lo; cx->hi = hi;
    pg_node *r = attach(v, 1, &a, bwd_clamp, cx, PG_AG_OP_CLAMP);
    if (!r) free(cx);
    return r;
}
pg_node *pg_ag_leaky_relu(pg_node *a, float alpha) {
    if (!a || !a->value) return NULL;
    pg_tensor *v = pg_leaky_relu(a->value, alpha);
    if (!v) return NULL;
    leaky_ctx_t *cx = malloc(sizeof(*cx));
    if (!cx) { pg_tensor_free(v); return NULL; }
    cx->alpha = alpha;
    pg_node *r = attach(v, 1, &a, bwd_leaky, cx, PG_AG_OP_LEAKY_RELU);
    if (!r) free(cx);
    return r;
}

pg_node *pg_ag_matmul(pg_node *a, pg_node *b)
{
    assert(a && b && a->value && b->value);
    pg_tensor *v = pg_matmul(a->value, b->value);
    if (!v)
        return NULL;
    return attach(v, 2, (pg_node *[]){a, b}, bwd_matmul, NULL, PG_AG_OP_MATMUL);
}

typedef struct { size_t axis0; size_t axis1; } transpose_ctx;

static void bwd_transpose(pg_node *n)
{
    transpose_ctx *cx = (transpose_ctx *)n->ctx;
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    pg_tensor *g = n->grad;
    size_t ndim = g->ndim;
    size_t shape[PG_MAX_NDIM];
    for (size_t i = 0; i < ndim; i++) {
        size_t axis = (i == cx->axis0) ? cx->axis1 : (i == cx->axis1 ? cx->axis0 : i);
        shape[i] = g->shape[axis];
    }
    pg_tensor *gt = pg_tensor_empty(ndim, shape);
    if (!gt) return;
    for (size_t i = 0; i < g->numel; i++) {
        size_t idx[PG_MAX_NDIM], idx_t[PG_MAX_NDIM];
        size_t tmp = i;
        for (size_t d = ndim; d-- > 0;) {
            idx[d] = tmp % g->shape[d];
            tmp /= g->shape[d];
        }
        for (size_t d = 0; d < ndim; d++) {
            size_t axis = (d == cx->axis0) ? cx->axis1 : (d == cx->axis1 ? cx->axis0 : d);
            idx_t[axis] = idx[d];
        }
        size_t off_t = 0, off_g = 0;
        for (size_t d = 0; d < ndim; d++) {
            off_t += idx_t[d] * gt->stride[d];
            off_g += idx[d] * g->stride[d];
        }
        gt->data[off_t] = g->data[off_g];
    }
    accum(p, gt, 1.0f);
    pg_tensor_free(gt);
}

pg_node *pg_ag_transpose(pg_node *a, size_t axis0, size_t axis1)
{
    assert(a && a->value);
    size_t ndim = a->value->ndim;
    assert(axis0 < ndim && axis1 < ndim);
    size_t nshape[PG_MAX_NDIM];
    for (size_t i = 0; i < ndim; i++)
        nshape[i] = a->value->shape[i];
    nshape[axis0] = a->value->shape[axis1];
    nshape[axis1] = a->value->shape[axis0];
    pg_tensor *v = pg_tensor_empty(ndim, nshape);
    if (!v)
        return NULL;
    const pg_tensor *src = a->value;
    for (size_t i = 0; i < src->numel; i++) {
        size_t idx[PG_MAX_NDIM], idx_t[PG_MAX_NDIM];
        size_t tmp = i;
        for (size_t d = ndim; d-- > 0;) {
            idx[d] = tmp % src->shape[d];
            tmp /= src->shape[d];
        }
        for (size_t d = 0; d < ndim; d++) {
            idx_t[d] = idx[(d == axis0) ? axis1 : (d == axis1) ? axis0 : d];
        }
        size_t off_t = 0, off_s = 0;
        for (size_t d = 0; d < ndim; d++) {
            off_t += idx_t[d] * v->stride[d];
            off_s += idx[d] * src->stride[d];
        }
        v->data[off_t] = src->data[off_s];
    }
    transpose_ctx *cx = malloc(sizeof(*cx));
    if (!cx) { pg_tensor_free(v); return NULL; }
    cx->axis0 = axis0; cx->axis1 = axis1;
    return attach(v, 1, &a, bwd_transpose, cx, PG_AG_OP_TRANSPOSE);
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

    pg_node *r = attach(v, 1, &a, mean ? bwd_mean : bwd_sum, cx,
                        mean ? PG_AG_OP_MEAN : PG_AG_OP_SUM);
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

    pg_node *r = attach(v, 1, &a, bwd_softmax, cx, PG_AG_OP_SOFTMAX);
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
        float sum_gw_x=0;
        for(size_t i=0;i<N;i++){
            float gw=grow[i]*(w?w[i]:1.0f);
            sum_gw_x += gw * row[i];
        }
        // gw = grad_y * w; dx = rstd*gw - x * (rstd^3 * invN * sum_gw_x)
        float coeff = - sum_gw_x * rstd * rstd * rstd * invN;
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
    pg_node *pars[3];
    size_t idx = 0;
    pars[idx++] = x;
    if (weight) pars[idx++] = weight;
    if (bias) pars[idx++] = bias;
    pg_node *r=attach(v, npar, pars, bwd_layernorm, cx, PG_AG_OP_LAYERNORM);
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
    pg_node *r=attach(v, npar, pars, bwd_rmsnorm, cx, PG_AG_OP_RMSNORM);
    if(!r) free(cx);
    return r;
}

// ---------- batchnorm ----------
typedef struct {
    float eps;
    size_t C;
    size_t perChannel;
    float *mean;
    float *invStd;
    bool has_w;
    bool has_b;
} bn_ctx_t;

static void bwd_batchnorm(pg_node *n){
    bn_ctx_t *cx = (bn_ctx_t*)n->ctx;
    size_t C = cx->C;
    size_t perCh = cx->perChannel;
    float *mean = cx->mean;
    float *invStd = cx->invStd;
    pg_node *px = n->parents[0];
    pg_node *pw = cx->has_w ? n->parents[1] : NULL;
    pg_node *pb = cx->has_b ? n->parents[cx->has_w ? 2 : 1] : NULL;
    const pg_tensor *x = px->value;
    const pg_tensor *g = n->grad;
    size_t ndim = x->ndim;

    // precompute inv per channel sums if needed
    // For weight/bias grads we need sum over perChannel
    if(pw && pw->requires_grad){
        ensure_grad(pw);
        // need to accumulate grad for weight: sum g * x_hat
        // compute per channel sums
        double *sum_g_xhat = calloc(C, sizeof(double));
        if(sum_g_xhat){
            size_t idx[PG_MAX_NDIM]={0};
            size_t off_x=0, off_g=0;
            for(size_t p=0;p<x->numel;p++){
                size_t c = idx[1];
                float xhat = (x->data[off_x] - mean[c]) * invStd[c];
                sum_g_xhat[c] += (double)g->data[off_g] * (double)xhat;
                for(size_t d=ndim; d-- >0;){
                    idx[d]++;
                    off_x += x->stride[d];
                    off_g += g->stride[d];
                    if(idx[d] < x->shape[d]) break;
                    idx[d]=0;
                    off_x -= x->stride[d] * x->shape[d];
                    off_g -= g->stride[d] * g->shape[d];
                }
            }
            for(size_t c=0;c<C;c++) pw->grad->data[c] += (float)sum_g_xhat[c];
            free(sum_g_xhat);
        }
    }
    if(pb && pb->requires_grad){
        ensure_grad(pb);
        double *sum_g = calloc(C, sizeof(double));
        if(sum_g){
            size_t idx[PG_MAX_NDIM]={0};
            size_t off_g=0;
            for(size_t p=0;p<g->numel;p++){
                size_t c = idx[1];
                sum_g[c] += (double)g->data[off_g];
                for(size_t d=ndim; d-- >0;){
                    idx[d]++;
                    off_g += g->stride[d];
                    if(idx[d] < x->shape[d]) break;
                    idx[d]=0;
                    off_g -= g->stride[d] * g->shape[d];
                }
            }
            for(size_t c=0;c<C;c++) pb->grad->data[c] += (float)sum_g[c];
            free(sum_g);
        }
    }
    if(!px->requires_grad) return;
    ensure_grad(px);
    // Compute per-channel sums needed for dx: mean(g) and mean(g*x_hat)
    double *sum_g = calloc(C, sizeof(double));
    double *sum_g_xhat = calloc(C, sizeof(double));
    if(!sum_g || !sum_g_xhat){ free(sum_g); free(sum_g_xhat); return; }
    size_t idx[PG_MAX_NDIM]={0};
    size_t off_x=0, off_g=0;
    for(size_t p=0;p<x->numel;p++){
        size_t c = idx[1];
        float xhat = (x->data[off_x] - mean[c]) * invStd[c];
        double gv = (double)g->data[off_g];
        sum_g[c] += gv;
        sum_g_xhat[c] += gv * (double)xhat;
        for(size_t d=ndim; d-- >0;){
            idx[d]++;
            off_x += x->stride[d];
            off_g += g->stride[d];
            if(idx[d] < x->shape[d]) break;
            idx[d]=0;
            off_x -= x->stride[d] * x->shape[d];
            off_g -= g->stride[d] * g->shape[d];
        }
    }
    // second pass: compute grad_x
    const float *w = (pw && pw->value) ? pw->value->data : NULL;
    float *gx = px->grad->data;
    memset(idx,0,sizeof(idx));
    off_x=0; off_g=0;
    size_t off_gx=0;
    // need gx stride (same as x likely)
    // we use px->grad stride which matches x shape
    for(size_t p=0;p<x->numel;p++){
        size_t c = idx[1];
        float xhat = (x->data[off_x] - mean[c]) * invStd[c];
        float gv = g->data[off_g];
        float mean_g = (float)(sum_g[c] / (double)perCh);
        float mean_gx = (float)(sum_g_xhat[c] / (double)perCh);
        float gamma = w ? w[c] : 1.0f;
        float dx = gamma * invStd[c] * (gv - mean_g - xhat * mean_gx);
        gx[off_gx] += dx;
        for(size_t d=ndim; d-- >0;){
            idx[d]++;
            off_x += x->stride[d];
            off_g += g->stride[d];
            off_gx += px->grad->stride[d];
            if(idx[d] < x->shape[d]) break;
            idx[d]=0;
            off_x -= x->stride[d] * x->shape[d];
            off_g -= g->stride[d] * g->shape[d];
            off_gx -= px->grad->stride[d] * x->shape[d];
        }
    }
    free(sum_g);
    free(sum_g_xhat);
}

pg_node *pg_ag_batchnorm2d(pg_node *x, pg_node *weight, pg_node *bias,
                           pg_tensor *running_mean, pg_tensor *running_var,
                           float eps, float momentum, bool training){
    assert(x && x->value);
    pg_tensor *v = pg_batchnorm2d(x->value, weight?weight->value:NULL, bias?bias->value:NULL,
                                  running_mean, running_var, eps, momentum, training);
    if(!v) return NULL;
    size_t C = x->value->shape[1];
    size_t perCh = x->value->numel / C;
    // Capture mean/invStd used for backward (if training, batch stats; else running)
    bn_ctx_t *cx = malloc(sizeof(*cx) + 2*C*sizeof(float));
    if(!cx){ pg_tensor_free(v); return NULL; }
    cx->eps = eps;
    cx->C = C;
    cx->perChannel = perCh;
    cx->has_w = weight != NULL;
    cx->has_b = bias != NULL;
    cx->mean = (float*)((char*)cx + sizeof(*cx));
    cx->invStd = cx->mean + C;
    if(training){
        // compute mean/invStd again for ctx (or copy from forward logic)
        // Recompute quickly to avoid storing forward intermediates
        double *mean_d = calloc(C, sizeof(double));
        double *var_d = calloc(C, sizeof(double));
        if (!mean_d || !var_d) {
            free(mean_d);
            free(var_d);
            free(cx);
            pg_tensor_free(v);
            return NULL;
        }
        size_t idx[PG_MAX_NDIM]={0};
        size_t off=0;
        for(size_t p=0;p<x->value->numel;p++){
            size_t c = idx[1];
            mean_d[c] += (double)x->value->data[off];
            for(size_t d=x->value->ndim; d-- >0;){
                idx[d]++;
                off += x->value->stride[d];
                if(idx[d] < x->value->shape[d]) break;
                idx[d]=0;
                off -= x->value->stride[d] * x->value->shape[d];
            }
        }
        for(size_t c=0;c<C;c++) mean_d[c] /= (double)perCh;
        memset(idx,0,sizeof(idx));
        off=0;
        for(size_t p=0;p<x->value->numel;p++){
            size_t c = idx[1];
            double d = (double)x->value->data[off] - mean_d[c];
            var_d[c] += d*d;
            for(size_t d2=x->value->ndim; d2-- >0;){
                idx[d2]++;
                off += x->value->stride[d2];
                if(idx[d2] < x->value->shape[d2]) break;
                idx[d2]=0;
                off -= x->value->stride[d2] * x->value->shape[d2];
            }
        }
        for(size_t c=0;c<C;c++){
            var_d[c] /= (double)perCh;
            cx->mean[c] = (float)mean_d[c];
            cx->invStd[c] = 1.0f / sqrtf((float)var_d[c] + eps);
        }
        free(mean_d); free(var_d);
    } else {
        // inference: use running stats
        for(size_t c=0;c<C;c++){
            cx->mean[c] = running_mean->data[c];
            cx->invStd[c] = 1.0f / sqrtf(running_var->data[c] + eps);
        }
    }
    size_t npar = 1 + (weight?1:0) + (bias?1:0);
    pg_node *pars[3];
    size_t pi = 0;
    pars[pi++] = x;
    if (weight) pars[pi++] = weight;
    if (bias) pars[pi++] = bias;
    // For inference mode, we still need grad for x/weight/bias if training=false but requires_grad, use same formula with running stats
    pg_node *r = attach(v, npar, pars, bwd_batchnorm, cx, PG_AG_OP_BATCHNORM);
    if(!r) free(cx);
    return r;
}

pg_node *pg_ag_batchnorm(pg_node *x, pg_node *weight, pg_node *bias, float eps){
    return pg_ag_batchnorm2d(x, weight, bias, NULL, NULL, eps, 0.1f, true);
}

// ---------- conv2d / embedding / dropout ----------

typedef struct { size_t kh, kw; int stride, padding; } conv_ctx;

static void bwd_conv(pg_node *n)
{
    conv_ctx *cx = (conv_ctx *)n->ctx;
    pg_node *px = n->parents[0];
    pg_node *pw = n->parents[1];
    pg_node *pb = n->nparents > 2 ? n->parents[2] : NULL;
    const pg_tensor *x = px->value;
    const pg_tensor *g = n->grad;
    size_t N = x->shape[0], Cin = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t Cout = pw->value->shape[0], kh = cx->kh, kw = cx->kw;
    int s = cx->stride, p = cx->padding;
    size_t OH = g->shape[2], OW = g->shape[3];

    if (pb && pb->requires_grad) {
        ensure_grad(pb);
        for (size_t co = 0; co < Cout; co++) {
            float acc = 0.0f;
            for (size_t ni = 0; ni < N; ni++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++)
                        acc += g->data[(ni * Cout + co) * OH * OW + oh * OW + ow];
            pb->grad->data[co] += acc;
        }
    }
    if (pw->requires_grad) {
        ensure_grad(pw);
        float *dw = pw->grad->data;
        for (size_t ni = 0; ni < N; ni++)
            for (size_t co = 0; co < Cout; co++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float gg = g->data[(ni * Cout + co) * OH * OW + oh * OW + ow];
                        if (gg == 0.0f)
                            continue;
                        for (size_t ci = 0; ci < Cin; ci++)
                            for (size_t i = 0; i < kh; i++)
                                for (size_t j = 0; j < kw; j++) {
                                    long ph = (long)oh * s - p + (long)i;
                                    long pw2 = (long)ow * s - p + (long)j;
                                    if (ph >= 0 && ph < (long)H && pw2 >= 0 && pw2 < (long)W) {
                                        size_t xidx = (ni * Cin + ci) * H * W +
                                                       (size_t)ph * W + (size_t)pw2;
                                        dw[((co * Cin + ci) * kh + i) * kw + j] += gg * x->data[xidx];
                                    }
                                }
                    }
    }
    if (px->requires_grad) {
        ensure_grad(px);
        float *dx = px->grad->data;
        const float *wv = pw->value->data;
        for (size_t ni = 0; ni < N; ni++)
            for (size_t co = 0; co < Cout; co++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float gg = g->data[(ni * Cout + co) * OH * OW + oh * OW + ow];
                        if (gg == 0.0f)
                            continue;
                        for (size_t ci = 0; ci < Cin; ci++)
                            for (size_t i = 0; i < kh; i++)
                                for (size_t j = 0; j < kw; j++) {
                                    long ph = (long)oh * s - p + (long)i;
                                    long pw2 = (long)ow * s - p + (long)j;
                                    if (ph >= 0 && ph < (long)H && pw2 >= 0 && pw2 < (long)W) {
                                        size_t xidx = (ni * Cin + ci) * H * W +
                                                       (size_t)ph * W + (size_t)pw2;
                                        dx[xidx] += gg * wv[((co * Cin + ci) * kh + i) * kw + j];
                                    }
                                }
                    }
    }
}

pg_node *pg_ag_conv2d(pg_node *x, pg_node *w, pg_node *b, size_t kh, size_t kw,
                      int stride, int padding)
{
    assert(x && w && x->value && w->value);
    pg_conv2d_cfg cfg = {stride, padding};
    pg_tensor *v = pg_conv2d(x->value, w->value, b ? b->value : NULL, cfg);
    if (!v)
        return NULL;
    conv_ctx *cx = malloc(sizeof(*cx));
    if (!cx) {
        pg_tensor_free(v);
        return NULL;
    }
    cx->kh = kh;
    cx->kw = kw;
    cx->stride = stride;
    cx->padding = padding;
    size_t np = b ? 3 : 2;
    pg_node *pars[3] = {x, w, b};
    pg_node *r = attach(v, np, pars, bwd_conv, cx, PG_AG_OP_CONV2D);
    if (!r)
        free(cx);
    return r;
}

typedef struct { size_t *idx; size_t n; size_t E; } emb_ctx;

static void bwd_embedding(pg_node *n)
{
    emb_ctx *cx = (emb_ctx *)n->ctx;
    pg_node *pw = n->parents[0];
    if (!pw->requires_grad)
        return;
    ensure_grad(pw);
    const pg_tensor *g = n->grad;
    float *dw = pw->grad->data;
    for (size_t i = 0; i < cx->n; i++) {
        size_t row = cx->idx[i];
        for (size_t e = 0; e < cx->E; e++)
            dw[row * cx->E + e] += g->data[i * cx->E + e];
    }
}

pg_node *pg_ag_embedding(pg_node *weight, const pg_tensor *indices)
{
    assert(weight && indices && weight->value && indices->data);
    assert(weight->value->ndim == 2);
    pg_tensor *v = pg_embedding(weight->value, indices);
    if (!v)
        return NULL;
    size_t n = indices->numel, E = weight->value->shape[1];
    emb_ctx *cx = malloc(sizeof(*cx) + n * sizeof(size_t));
    if (!cx) {
        pg_tensor_free(v);
        return NULL;
    }
    cx->n = n;
    cx->E = E;
    cx->idx = (size_t *)((char *)cx + sizeof(*cx));
    for (size_t i = 0; i < n; i++)
        cx->idx[i] = (size_t)indices->data[i];
    pg_node *r = attach(v, 1, &weight, bwd_embedding, cx, PG_AG_OP_EMBEDDING);
    if (!r)
        free(cx);
    return r;
}

static uint64_t pg_drop_rng = 0x9E3779B97F4A7C15ULL;
static float pg_drop_rand(void)
{
    uint64_t z = (pg_drop_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

typedef struct { float *mask; size_t n; float scale; bool training; } drop_ctx;

static void bwd_dropout(pg_node *n)
{
    drop_ctx *cx = (drop_ctx *)n->ctx;
    pg_node *px = n->parents[0];
    if (!px->requires_grad)
        return;
    ensure_grad(px);
    const pg_tensor *g = n->grad;
    float *dx = px->grad->data;
    if (!cx->training) {
        for (size_t i = 0; i < cx->n; i++)
            dx[i] += g->data[i];
    } else {
        for (size_t i = 0; i < cx->n; i++)
            dx[i] += g->data[i] * cx->mask[i] * cx->scale;
    }
}

pg_tensor *pg_dropout(const pg_tensor *x, float p, bool training)
{
    assert(x && x->data);
    if (!training || p <= 0.0f)
        return pg_tensor_clone(x);
    pg_tensor *out = pg_tensor_empty(x->ndim, x->shape);
    if (!out)
        return NULL;
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x->numel; i++) {
        float r = pg_drop_rand();
        if (r < p)
            out->data[i] = 0.0f;
        else
            out->data[i] = x->data[i] * scale;
    }
    return out;
}

pg_node *pg_ag_dropout(pg_node *x, float p, bool training)
{
    assert(x && x->value);
    float scale = (p < 1.0f) ? 1.0f / (1.0f - p) : 1.0f;
    pg_tensor *v = pg_tensor_empty(x->value->ndim, x->value->shape);
    if (!v)
        return NULL;
    drop_ctx *cx = malloc(sizeof(*cx) + x->value->numel * sizeof(float));
    if (!cx) {
        pg_tensor_free(v);
        return NULL;
    }
    cx->n = x->value->numel;
    cx->scale = scale;
    cx->training = training && p > 0.0f;
    cx->mask = (float *)((char *)cx + sizeof(*cx));
    if (cx->training) {
        for (size_t i = 0; i < cx->n; i++) {
            float r = pg_drop_rand();
            if (r < p) {
                cx->mask[i] = 0.0f;
                v->data[i] = 0.0f;
            } else {
                cx->mask[i] = 1.0f;
                v->data[i] = x->value->data[i] * scale;
            }
        }
    } else {
        for (size_t i = 0; i < cx->n; i++) {
            cx->mask[i] = 1.0f;
            v->data[i] = x->value->data[i];
        }
    }
    pg_node *r = attach(v, 1, &x, bwd_dropout, cx, PG_AG_OP_DROPOUT);
    if (!r)
        free(cx);
    return r;
}

// ---------- loss functions (training) ----------

typedef struct {
    size_t rows;
    size_t cls;
    bool mean;
    size_t *targets;
} ce_ctx;

static void bwd_cross_entropy(pg_node *n)
{
    ce_ctx *cx = n->ctx;
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    const pg_tensor *L = p->value;
    size_t R = cx->rows, C = cx->cls;
    pg_tensor *g = pg_tensor_zeros(L->ndim, L->shape);
    if (!g)
        return;
    float scale = cx->mean ? 1.0f / (float)R : 1.0f;
    for (size_t r = 0; r < R; r++) {
        size_t base = r * C;
        float m = -INFINITY;
        for (size_t c = 0; c < C; c++)
            if (L->data[base + c] > m)
                m = L->data[base + c];
        float s = 0.0f;
        for (size_t c = 0; c < C; c++)
            s += expf(L->data[base + c] - m);
        float inv = 1.0f / s;
        for (size_t c = 0; c < C; c++) {
            float prob = expf(L->data[base + c] - m) * inv;
            float onehot = (c == cx->targets[r]) ? 1.0f : 0.0f;
            g->data[base + c] = (prob - onehot) * scale;
        }
    }
    accum(p, g, 1.0f);
    pg_tensor_free(g);
}

pg_node *pg_ag_cross_entropy(pg_node *logits, const size_t *targets, size_t n, bool mean)
{
    assert(logits && logits->value && targets);
    pg_tensor *L = logits->value;
    size_t cls = L->shape[L->ndim - 1];
    size_t rows = L->numel / cls;
    assert(rows == n);
#ifndef NDEBUG
    for (size_t r = 0; r < n; r++)
        assert(targets[r] < cls);
#endif

    float total = 0.0f;
    for (size_t r = 0; r < rows; r++) {
        size_t base = r * cls;
        float m = -INFINITY;
        for (size_t c = 0; c < cls; c++)
            if (L->data[base + c] > m)
                m = L->data[base + c];
        float s = 0.0f;
        for (size_t c = 0; c < cls; c++)
            s += expf(L->data[base + c] - m);
        float lse = logf(s) + m;
        total += lse - L->data[base + targets[r]];
    }
    float loss = mean ? total / (float)rows : total;

    pg_tensor *v = pg_tensor_full(1, (size_t[]){1}, loss);
    if (!v)
        return NULL;
    ce_ctx *cx = malloc(sizeof(*cx) + n * sizeof(size_t));
    if (!cx) {
        pg_tensor_free(v);
        return NULL;
    }
    cx->rows = rows;
    cx->cls = cls;
    cx->mean = mean;
    cx->targets = (size_t *)((char *)cx + sizeof(*cx));
    memcpy(cx->targets, targets, n * sizeof(size_t));

    pg_node *r = attach(v, 1, &logits, bwd_cross_entropy, cx, PG_AG_OP_CROSS_ENTROPY);
    if (!r) {
        free(cx);
    }
    return r;
}

pg_node *pg_ag_mse(pg_node *pred, pg_node *target, bool mean)
{
    assert(pred && target);
    pg_node *d = pg_ag_sub(pred, target);
    if (!d)
        return NULL;
    pg_node *sq = pg_ag_mul(d, d);
    pg_node_free(d);
    if (!sq)
        return NULL;
    pg_node *loss = mean ? pg_ag_mean_all(sq) : pg_ag_sum_all(sq);
    pg_node_free(sq);
    return loss;
}

typedef struct {
    size_t n;
    bool mean;
    float *targets;
} bce_ctx;

static void bwd_bce(pg_node *n)
{
    bce_ctx *cx = n->ctx;
    pg_node *p = n->parents[0];
    if (!p->requires_grad)
        return;
    const pg_tensor *L = p->value;
    size_t N = cx->n;
    float scale = cx->mean ? 1.0f / (float)N : 1.0f;
    pg_tensor *g = pg_tensor_zeros(L->ndim, L->shape);
    if (!g)
        return;
    for (size_t i = 0; i < N; i++) {
        float z = L->data[i];
        float s = 1.0f / (1.0f + expf(-z));
        g->data[i] = (s - cx->targets[i]) * scale;
    }
    accum(p, g, 1.0f);
    pg_tensor_free(g);
}

pg_node *pg_ag_bce_with_logits(pg_node *logits, const float *targets, size_t n, bool mean)
{
    assert(logits && logits->value && targets);
    const pg_tensor *L = logits->value;
    assert(L->numel == n);

    float total = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float z = L->data[i];
        float t = targets[i];
        total += fmaxf(z, 0.0f) - z * t + log1pf(expf(-fabsf(z)));
    }
    float loss = mean ? total / (float)n : total;

    pg_tensor *v = pg_tensor_full(1, (size_t[]){1}, loss);
    if (!v)
        return NULL;
    bce_ctx *cx = malloc(sizeof(*cx) + n * sizeof(float));
    if (!cx) {
        pg_tensor_free(v);
        return NULL;
    }
    cx->n = n;
    cx->mean = mean;
    cx->targets = (float *)((char *)cx + sizeof(*cx));
    memcpy(cx->targets, targets, n * sizeof(float));

    pg_node *r = attach(v, 1, &logits, bwd_bce, cx, PG_AG_OP_BCE);
    if (!r) {
        free(cx);
    }
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
        if (!ni) {
            assert(ni);
            return;
        }
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
    {
        size_t ncap = cap ? cap * 2 : 32;
        if (sp == cap) {
            frame *ns = malloc(ncap * sizeof(*ns));
            if (!ns) {
                assert(ns);
                return;
            }
            if (stk) {
                memcpy(ns, stk, sp * sizeof(*ns));
                free(stk);
            }
            stk = ns;
            cap = ncap;
        }
        stk[sp].n = root;
        stk[sp].i = 0;
        sp++;
    }

    while (sp) {
        frame *f = &stk[sp - 1];
        if (f->i < f->n->nparents) {
            pg_node *c = f->n->parents[f->i++];
            if (c->requires_grad && !c->mark) {
                c->mark = 1;
                if (sp == cap) {
                    size_t ncap = cap ? cap * 2 : 32;
                    frame *ns = malloc(ncap * sizeof(*ns));
                    if (!ns) {
                        assert(ns);
                        break;
                    }
                    memcpy(ns, stk, sp * sizeof(*ns));
                    free(stk);
                    stk = ns;
                    cap = ncap;
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

/* ---------- JIT autograd helpers ---------- */
static int find_order_idx(node_vec *order, pg_node *n) {
    for (size_t i = 0; i < order->n; i++) if (order->items[i] == n) return (int)i;
    return -1;
}

/* map from pg_node* to jit input id for saved values */
typedef struct { pg_node *node; int jit_id; } val_map_t;

static int jit_get_val_input(pg_jit_graph *jg, pg_node *p, val_map_t **maps, size_t *nmaps, size_t *cap) {
    for (size_t i = 0; i < *nmaps; i++) if ((*maps)[i].node == p) return (*maps)[i].jit_id;
    if (*nmaps == *cap) {
        size_t nc = *cap ? *cap * 2 : 8;
        val_map_t *nm = realloc(*maps, nc * sizeof(*nm));
        if (!nm) return -1;
        *maps = nm; *cap = nc;
    }
    int id = pg_jit_add_input(jg, p->value->ndim, p->value->shape);
    if (id < 0) return -1;
    (*maps)[*nmaps].node = p;
    (*maps)[*nmaps].jit_id = id;
    (*nmaps)++;
    return id;
}

static bool ag_needs_val_for_parent(pg_ag_op_t op, size_t parent_idx) {
    switch (op) {
        case PG_AG_OP_ADD: case PG_AG_OP_SUB: case PG_AG_OP_NEG: return false;
        case PG_AG_OP_MUL: case PG_AG_OP_DIV: return true;
        case PG_AG_OP_EXP: case PG_AG_OP_LOG: case PG_AG_OP_SQRT:
        case PG_AG_OP_SIN: case PG_AG_OP_COS: case PG_AG_OP_RELU:
        case PG_AG_OP_SIGMOID: case PG_AG_OP_TANH: return parent_idx == 0;
        default: return false;
    }
}
static bool is_bcast_compat(size_t ndim_a, const size_t *shape_a, size_t ndim_b, const size_t *shape_b) {
    if (ndim_a > ndim_b) return false;
    size_t off = ndim_b - ndim_a;
    for (size_t i=0;i<ndim_b;i++) {
        size_t da = i < off ? 1 : shape_a[i - off];
        size_t db = shape_b[i];
        if (da != db && da != 1) return false;
    }
    return true;
}

static bool try_jit_backward_internal_ex(pg_node *loss, node_vec *order, bool require_full);

static __attribute__((unused)) bool try_jit_backward_internal_full(pg_node *loss, node_vec *order) {
    return try_jit_backward_internal_ex(loss, order, true);
}
static bool try_jit_backward_internal(pg_node *loss, node_vec *order) {
    return try_jit_backward_internal_ex(loss, order, false);
}
static bool try_jit_backward_internal_ex(pg_node *loss, node_vec *order, bool require_full) {
    if (!loss || !order || order->n == 0) { ag_set_err("empty order"); return false; }
    /* step 1: handle reduction prefix (sum/mean chain) leading to scalar loss */
    pg_node *eff_loss = loss;
    float red_scale = 1.0f;
    size_t red_steps = 0;
    pg_node *prefix_nodes[32];
    size_t n_prefix = 0;
    while (eff_loss && (eff_loss->ag_op == PG_AG_OP_SUM || eff_loss->ag_op == PG_AG_OP_MEAN)) {
        if (n_prefix >= 32) { ag_set_err("too many reductions"); return false; }
        prefix_nodes[n_prefix++] = eff_loss;
        if (eff_loss->ag_op == PG_AG_OP_MEAN) {
            red_ctx *rc = (red_ctx*)eff_loss->ctx;
            if (!eff_loss->nparents || !eff_loss->parents[0] || !eff_loss->parents[0]->value) {
                ag_set_err("mean parent missing");
                return false;
            }
            size_t axis = rc->axis;
            if (axis >= eff_loss->parents[0]->value->ndim) { ag_set_err("mean axis oob"); return false; }
            size_t dim = eff_loss->parents[0]->value->shape[axis];
            if (dim == 0) { ag_set_err("mean dim 0"); return false; }
            red_scale *= 1.0f / (float)dim;
        }
        if (eff_loss->nparents == 0) break;
        eff_loss = eff_loss->parents[0];
        if (++red_steps > 32) { ag_set_err("red loop"); return false; }
        if (eff_loss == loss) { ag_set_err("cycle"); return false; }
    }
    if (!eff_loss) { ag_set_err("no eff loss"); return false; }
    if (eff_loss->ag_op == PG_AG_OP_NONE) {
        ag_set_err("no elementwise eff_loss");
        return false;
    }
    if (!ag_op_is_jit_elementwise(eff_loss->ag_op)) {
        ag_set_err("eff_loss not jit elementwise %d", eff_loss->ag_op);
        return false;
    }
    size_t loop_ndim = eff_loss->value->ndim;
    size_t loop_shape[PG_MAX_NDIM];
    memcpy(loop_shape, eff_loss->value->shape, loop_ndim * sizeof(size_t));
    size_t loop_numel = eff_loss->value->numel;

    /* find suffix of elementwise ops with same loop_shape reachable from eff_loss */
    bool *in_suffix = calloc(order->n, sizeof(bool));
    int *queue = malloc(order->n * sizeof(int));
    if (!in_suffix || !queue) {
        free(in_suffix);
        free(queue);
        ag_set_err("oom suffix");
        return false;
    }
    int eff_idx = find_order_idx(order, eff_loss);
    if (eff_idx < 0) {
        free(in_suffix);
        free(queue);
        ag_set_err("eff idx not found");
        return false;
    }
    size_t qh=0, qt=0;
    queue[qt++] = eff_idx;
    in_suffix[eff_idx]=true;
    while (qh < qt) {
        int idx = queue[qh++];
        pg_node *n = order->items[idx];
        for (size_t pi=0; pi<n->nparents; pi++) {
            pg_node *par = n->parents[pi];
            int p_idx = find_order_idx(order, par);
            if (p_idx < 0) continue;
            if (in_suffix[p_idx]) continue;
            pg_node *pn = order->items[p_idx];
            if (pn->ag_op == PG_AG_OP_NONE) continue;
            if (!ag_op_is_jit_elementwise(pn->ag_op)) continue;
            if (pn->value->ndim != loop_ndim || pn->value->numel != loop_numel ||
                !pg_shape_equal(pn->value->ndim, pn->value->shape, loop_ndim, loop_shape))
                continue;
            in_suffix[p_idx]=true;
            queue[qt++]=p_idx;
        }
    }
    free(queue);
    /* count suffix nodes */
    size_t n_suffix = 0;
    for (size_t i = 0; i < order->n; i++) if (in_suffix[i]) n_suffix++;
    if (n_suffix == 0) {
        free(in_suffix);
        ag_set_err("no suffix");
        return false;
    }

    /* check that all suffix nodes have same loop_shape and needed parent values have same shape */
    for (size_t i = 0; i < order->n; i++) if (in_suffix[i]) {
        pg_node *n = order->items[i];
        if (n->value->ndim != loop_ndim || n->value->numel != loop_numel ||
            !pg_shape_equal(n->value->ndim, n->value->shape, loop_ndim, loop_shape)) {
                free(in_suffix);
                ag_set_err("suffix shape mismatch");
                return false;
            }
        for (size_t pi=0; pi<n->nparents; pi++) {
            if (!ag_needs_val_for_parent(n->ag_op, pi)) continue;
            pg_node *par = n->parents[pi];
            if (!par->value) { free(in_suffix); ag_set_err("parent value null"); return false; }
            if (!is_bcast_compat(par->value->ndim, par->value->shape, loop_ndim, loop_shape)) {
                free(in_suffix); ag_set_err("parent value shape mismatch for suffix"); return false;
            }
        }
    }

    /* collect suffix leaf inputs that require grad but are not in suffix (these will be outputs of JIT) */
    /* also collect leaf vars in suffix that are leaves */
    /* For suffix JIT, outputs are grads for nodes that are parents of suffix nodes but not in suffix, plus suffix leaves */
    bool *is_suffix_output = calloc(order->n, sizeof(bool));
    if (!is_suffix_output) { free(in_suffix); ag_set_err("oom"); return false; }
    // First, for each suffix node, its parents that require_grad and are not in suffix will need grad output
    for (size_t i=0;i<order->n;i++) if(in_suffix[i]) {
        pg_node *n = order->items[i];
        for (size_t pi=0; pi<n->nparents; pi++) {
            pg_node *par = n->parents[pi];
            if (!par->requires_grad) continue;
            int p_idx = find_order_idx(order, par);
            if (p_idx>=0 && in_suffix[p_idx]) continue; // parent also in suffix, its grad will be computed internally, not output yet
            // parent not in suffix, but requires grad -> need to output its grad
            if (p_idx>=0) is_suffix_output[p_idx]=true;
            else {
                // parent not in order (maybe not requires_grad? but we check requires_grad, so it should be in order)
                // If parent not in order but requires_grad, it means it's a leaf not in order due to topo not including? Actually topo includes all requires_grad ancestors, so it should be in order.
                // If not found, we need to handle as leaf input: we can still output, but we need to find its index? It's not in order, so we can't use order index. For now, treat as not output.
            }
        }
    }
    // Also, if suffix contains leaf vars (nparents==0) that require_grad, they are outputs
    for (size_t i=0;i<order->n;i++) if(in_suffix[i]) {
        pg_node *n = order->items[i];
        if (n->nparents==0 && n->requires_grad) is_suffix_output[i]=true;
    }

    size_t n_out=0;
    for (size_t i=0;i<order->n;i++) if(is_suffix_output[i]) n_out++;
    if (n_out == 0) {
        free(in_suffix);
        free(is_suffix_output);
        ag_set_err("no suffix outputs");
        return false;
    }
    if (n_out > 16) {
        free(in_suffix);
        free(is_suffix_output);
        ag_set_err("too many suffix outputs");
        return false;
    }

    pg_jit_graph *jg = pg_jit_graph_new();
    if (!jg) {
        free(in_suffix);
        free(is_suffix_output);
        ag_set_err("jg alloc fail");
        return false;
    }

    val_map_t *val_maps = NULL;
    size_t n_val_maps = 0, cap_val_maps = 0;
    int *grad_jit_id = calloc(order->n, sizeof(int));
    if (!grad_jit_id) {
        pg_jit_graph_free(jg);
        free(in_suffix);
        free(is_suffix_output);
        free(val_maps);
        ag_set_err("oom grad map");
        return false;
    }
    for (size_t i = 0; i < order->n; i++) grad_jit_id[i] = -1;

    pg_tensor *g_eff_tensor = NULL;
    pg_tensor *leaf_tmp[16] = {0};
    int g_eff = pg_jit_add_input(jg, loop_ndim, loop_shape);
    if (g_eff < 0) { ag_set_err("g_eff input fail %s", pg_jit_last_error()); goto jit_fail; }
    grad_jit_id[eff_idx] = g_eff;

    for (size_t i = 0; i < order->n; i++) if(in_suffix[i]) {
        pg_node *n = order->items[i];
        for (size_t pi = 0; pi < n->nparents; pi++) {
            if (!ag_needs_val_for_parent(n->ag_op, pi)) continue;
            pg_node *par = n->parents[pi];
            int vid = jit_get_val_input(jg, par, &val_maps, &n_val_maps, &cap_val_maps);
            if (vid < 0) { ag_set_err("val input fail %s", pg_jit_last_error()); goto jit_fail; }
        }
    }

    #define GET_VAL_JIT(par) ({ \
        int _vid = -1; \
        for (size_t _k = 0; _k < n_val_maps; _k++) if (val_maps[_k].node == (par)) { _vid = val_maps[_k].jit_id; break; } \
        _vid; \
    })

    /* iterate reverse over suffix nodes in topo order (from eff down to leaves) */
    // order is forward leaf->loss, so suffix nodes are somewhere in middle. Reverse suffix order is from eff_idx down to 0, but only those in suffix.
    // We should iterate order in reverse, and if in_suffix, process.
    for (int oi = (int)order->n - 1; oi >= 0; --oi) {
        if (!in_suffix[oi]) continue;
        pg_node *n = order->items[oi];
        int g_child = grad_jit_id[oi];
        if (g_child < 0) continue;
        for (size_t pi = 0; pi < n->nparents; pi++) {
            pg_node *par = n->parents[pi];
            if (!par->requires_grad) continue;
            int p_idx = find_order_idx(order, par);
            // par may not be in order if it's a leaf not requiring grad? But we check requires_grad, so it should be.
            // If par not in suffix, its grad is an output, but we still need to compute it. p_idx may be -1 if par is not in order (e.g., leaf bias not in order because not requires_grad? but we check requires_grad, so it should be)
            // For suffix inputs that are not in suffix, p_idx may be valid but in_suffix false, we still need to accumulate into that parent's grad.
            // If p_idx <0, we need to handle as external leaf not in order: we could allocate a separate grad for it, but our grad_jit_id is only for order indices. For external leaf not in order, we need a separate handling.
            // For now, require p_idx >=0
            if (p_idx < 0) continue;
            // If parent is not in suffix, its grad may be an output, but we still compute contrib for it.
            // The parent's grad may already have a value from previous contributions (multiple children in suffix)
            int contrib = -1;
            int val_a = -1, val_b = -1;
            if (n->ag_op == PG_AG_OP_MUL || n->ag_op == PG_AG_OP_DIV) {
                val_a = GET_VAL_JIT(n->parents[0]);
                val_b = GET_VAL_JIT(n->parents[1]);
            } else if (n->ag_op == PG_AG_OP_EXP || n->ag_op == PG_AG_OP_LOG || n->ag_op == PG_AG_OP_SQRT ||
                       n->ag_op == PG_AG_OP_SIN || n->ag_op == PG_AG_OP_COS || n->ag_op == PG_AG_OP_RELU ||
                       n->ag_op == PG_AG_OP_SIGMOID || n->ag_op == PG_AG_OP_TANH) {
                val_a = GET_VAL_JIT(n->parents[0]);
            }
            switch (n->ag_op) {
                case PG_AG_OP_ADD: { contrib = g_child; break; }
                case PG_AG_OP_SUB: {
                    if (pi == 0) contrib = g_child;
                    else {
                        contrib = pg_jit_neg(jg, g_child);
                        if (contrib < 0) {
                            ag_set_err("neg fail");
                            goto jit_fail;
                        }
                    }
                    break;
                }
                case PG_AG_OP_MUL: {
                    int other_val = (pi == 0) ? val_b : val_a;
                    if (other_val < 0) { ag_set_err("mul val missing"); goto jit_fail; }
                    contrib = pg_jit_mul(jg, g_child, other_val);
                    if (contrib < 0) { ag_set_err("mul fail %s", pg_jit_last_error()); goto jit_fail; }
                    break;
                }
                case PG_AG_OP_DIV: {
                    if (pi == 0) {
                        int vb = val_b; if (vb < 0) { ag_set_err("div vb missing"); goto jit_fail; }
                        contrib = pg_jit_div(jg, g_child, vb); if (contrib < 0) { ag_set_err("div fail"); goto jit_fail; }
                    } else {
                        int va = val_a;
                        int vb = val_b;
                        if (va < 0 || vb < 0) {
                            ag_set_err("div vals missing");
                            goto jit_fail;
                        }
                        int vb2 = pg_jit_mul(jg, vb, vb); if (vb2 < 0) goto jit_fail;
                        int num = pg_jit_mul(jg, g_child, va); if (num < 0) goto jit_fail;
                        int div_tmp = pg_jit_div(jg, num, vb2); if (div_tmp < 0) goto jit_fail;
                        contrib = pg_jit_neg(jg, div_tmp); if (contrib < 0) goto jit_fail;
                    }
                    break;
                }
                case PG_AG_OP_NEG: {
                    contrib = pg_jit_neg(jg, g_child);
                    if (contrib < 0) goto jit_fail;
                    break;
                }
                case PG_AG_OP_EXP: {
                    if (val_a < 0) { ag_set_err("exp val missing"); goto jit_fail; }
                    int exp_a = pg_jit_exp(jg, val_a); if (exp_a < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, exp_a); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_LOG: {
                    if (val_a < 0) goto jit_fail;
                    contrib = pg_jit_div(jg, g_child, val_a); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_SQRT: {
                    if (val_a < 0) goto jit_fail;
                    int sqrt_a = pg_jit_sqrt(jg, val_a); if (sqrt_a < 0) goto jit_fail;
                    int half = pg_jit_add_const(jg, 0.5f); if (half < 0) goto jit_fail;
                    int inv = pg_jit_div(jg, half, sqrt_a); if (inv < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, inv); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_SIN: {
                    if (val_a < 0) goto jit_fail;
                    int cos_a = pg_jit_cos(jg, val_a); if (cos_a < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, cos_a); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_COS: {
                    if (val_a < 0) goto jit_fail;
                    int sin_a = pg_jit_sin(jg, val_a); if (sin_a < 0) goto jit_fail;
                    int neg_sin = pg_jit_neg(jg, sin_a); if (neg_sin < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, neg_sin); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_RELU: {
                    if (val_a < 0) goto jit_fail;
                    int step = pg_jit_step(jg, val_a); if (step < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, step); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_SIGMOID: {
                    if (val_a < 0) goto jit_fail;
                    int s = pg_jit_sigmoid(jg, val_a); if (s < 0) goto jit_fail;
                    int one = pg_jit_add_const(jg, 1.0f); if (one < 0) goto jit_fail;
                    int one_minus_s = pg_jit_sub(jg, one, s); if (one_minus_s < 0) goto jit_fail;
                    int local = pg_jit_mul(jg, s, one_minus_s); if (local < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, local); if (contrib < 0) goto jit_fail; break;
                }
                case PG_AG_OP_TANH: {
                    if (val_a < 0) goto jit_fail;
                    int t = pg_jit_tanh(jg, val_a); if (t < 0) goto jit_fail;
                    int t2 = pg_jit_mul(jg, t, t); if (t2 < 0) goto jit_fail;
                    int one = pg_jit_add_const(jg, 1.0f); if (one < 0) goto jit_fail;
                    int local = pg_jit_sub(jg, one, t2); if (local < 0) goto jit_fail;
                    contrib = pg_jit_mul(jg, g_child, local); if (contrib < 0) goto jit_fail; break;
                }
                default: { ag_set_err("unhandled op %d", n->ag_op); goto jit_fail; }
            }
            if (contrib < 0) { ag_set_err("contrib fail"); goto jit_fail; }
            // p_idx may be in suffix or not. If in suffix, its grad is internal; if not, it's output.
            // For output case, we still store in grad_jit_id for that parent index to allow accumulation if multiple suffix children contribute to same parent.
            if (grad_jit_id[p_idx] < 0) {
                grad_jit_id[p_idx] = contrib;
            } else {
                int added = pg_jit_add(jg, grad_jit_id[p_idx], contrib);
                if (added < 0) { ag_set_err("add accum fail"); goto jit_fail; }
                grad_jit_id[p_idx] = added;
            }
        }
    }
    #undef GET_VAL_JIT

    int leaf_out_ids[16];
    pg_node *leaf_nodes[16];
    bool leaf_is_bcast[16] = {0};
    size_t n_leaf = 0;
    for (size_t i = 0; i < order->n; i++) if(is_suffix_output[i]) {
        int gid = grad_jit_id[i];
        if (gid < 0) continue;
        if (n_leaf >= 16) { ag_set_err("too many leaf outs"); goto jit_fail; }
        pg_node *ln = order->items[i];
        if (!is_bcast_compat(ln->value->ndim, ln->value->shape, loop_ndim, loop_shape)) {
            ag_set_err("leaf output shape not bcast compat");
            goto jit_fail;
        }
        bool is_bcast = !pg_shape_equal(ln->value->ndim, ln->value->shape, loop_ndim, loop_shape);
        leaf_is_bcast[n_leaf] = is_bcast;
        leaf_out_ids[n_leaf] = gid;
        leaf_nodes[n_leaf] = ln;
        n_leaf++;
    }
    if (n_leaf == 0) { ag_set_err("no leaf grads to output"); goto jit_fail; }
    for (size_t i = 0; i < n_leaf; i++) pg_jit_mark_output(jg, leaf_out_ids[i]);

    pg_jit_exe *exe = pg_jit_compile(jg);
    if (!exe) { ag_set_err("compile fail %s", pg_jit_last_error()); goto jit_fail; }

    /* prepare run: zero grads for suffix outputs (and also for all? we will zero all grads in pg_backward before, but for suffix we need to ensure) */
    for (size_t i = 0; i < order->n; i++) {
        pg_node *n = order->items[i];
        if (is_suffix_output[i]) {
            ensure_grad(n);
            if (!n->grad) { ag_set_err("ensure grad fail"); pg_jit_exe_free(exe); goto jit_fail; }
            // zero will be done by pg_backward before, but ensure zero
        }
    }

    g_eff_tensor = pg_tensor_full(loop_ndim, loop_shape, red_scale);
    if (!g_eff_tensor) { ag_set_err("g_eff tensor alloc fail"); goto jit_fail; }
    const pg_tensor *jit_inputs[16];
    pg_tensor *jit_outputs[16];
    // g_eff is first input (index 0), val_maps inputs follow
    jit_inputs[0] = g_eff_tensor;
    for (size_t i = 0; i < n_val_maps; i++) {
        jit_inputs[1+i] = val_maps[i].node->value;
    }
    for (size_t i = 0; i < n_leaf; i++) {
        if (leaf_is_bcast[i]) {
            pg_tensor *tmp = pg_tensor_zeros(loop_ndim, loop_shape);
            if (!tmp) { ag_set_err("tmp alloc fail"); pg_jit_exe_free(exe); goto jit_fail; }
            leaf_tmp[i] = tmp;
            jit_outputs[i] = tmp;
        } else {
            jit_outputs[i] = leaf_nodes[i]->grad;
        }
    }

    bool ok = pg_jit_run(exe, jit_inputs, n_val_maps+1, jit_outputs, n_leaf);
    pg_tensor_free(g_eff_tensor);
    if (ok) {
        for (size_t i=0;i<n_leaf;i++) if(leaf_is_bcast[i]) {
            accum(leaf_nodes[i], leaf_tmp[i], 1.0f);
            pg_tensor_free(leaf_tmp[i]);
        }
    } else {
        for (size_t i=0;i<n_leaf;i++) if(leaf_tmp[i]) pg_tensor_free(leaf_tmp[i]);
    }
    pg_jit_exe_free(exe);
    pg_jit_graph_free(jg);
    free(val_maps);
    free(grad_jit_id);
    // if require_full, ensure all non-prefix elementwise nodes were in suffix
    if (require_full) {
        for (size_t i=0;i<order->n;i++) {
            bool is_prefix = false;
            for (size_t pp=0; pp<n_prefix; pp++) if (prefix_nodes[pp]==order->items[i]) {is_prefix=true; break;}
            if (is_prefix) continue;
            pg_node *nn = order->items[i];
            if (nn->ag_op==PG_AG_OP_NONE) continue;
            if (!in_suffix[i]) {
                ag_set_err("not all nodes in suffix for full");
                free(in_suffix);
                free(is_suffix_output);
                return false;
            }
        }
        free(in_suffix);
        free(is_suffix_output);
    } else {
        free(in_suffix);
        free(is_suffix_output);
    }
    if (!ok) { ag_set_err("jit run fail %s", pg_jit_last_error()); return false; }
    return true;

jit_fail:
    pg_jit_graph_free(jg);
    free(val_maps);
    free(grad_jit_id);
    free(in_suffix);
    free(is_suffix_output);
    if (g_eff_tensor) pg_tensor_free(g_eff_tensor);
    for (size_t _lt = 0; _lt < 16; _lt++) if (leaf_tmp[_lt]) pg_tensor_free(leaf_tmp[_lt]);
    return false;
}



static bool try_jit_suffix(pg_node *loss, node_vec *order, bool **out_handled) {
    // Use the internal suffix JIT (require_full=false) and if it succeeds, compute in_suffix for caller
    bool ok = try_jit_backward_internal_ex(loss, order, false);
    if (!ok) return false;
    // Recompute in_suffix for hybrid handling (same logic as inside try_jit)
    pg_node *eff_loss = loss;
    size_t n_prefix = 0;
    while (eff_loss && (eff_loss->ag_op == PG_AG_OP_SUM || eff_loss->ag_op == PG_AG_OP_MEAN)) {
        if (n_prefix >= 32) return true; // suffix was handled, but we can't compute in_suffix precisely, return without handled
        n_prefix++;
        if (eff_loss->nparents == 0) break;
        eff_loss = eff_loss->parents[0];
    }
    if (!eff_loss || eff_loss->ag_op==PG_AG_OP_NONE || !ag_op_is_jit_elementwise(eff_loss->ag_op)) {
        // suffix was empty? Still return true but no handled
        bool *h = calloc(order->n, sizeof(bool));
        if (!h && order->n != 0) {
            *out_handled = NULL;
            return true;
        }
        *out_handled = h;
        return true;
    }
    size_t loop_ndim = eff_loss->value->ndim;
    size_t loop_shape[PG_MAX_NDIM]; memcpy(loop_shape, eff_loss->value->shape, loop_ndim*sizeof(size_t));
    size_t loop_numel = eff_loss->value->numel;
    bool *in_suffix = calloc(order->n, sizeof(bool));
    int *queue = malloc(order->n * sizeof(int));
    if (!in_suffix || !queue) {
        free(in_suffix);
        free(queue);
        bool *h = calloc(order->n, sizeof(bool));
        if (!h && order->n != 0) {
            *out_handled = NULL;
            return true;
        }
        *out_handled = h;
        return true;
    }
    int eff_idx = find_order_idx(order, eff_loss);
    if (eff_idx>=0) {
        size_t qh=0, qt=0;
        queue[qt++]=eff_idx; in_suffix[eff_idx]=true;
        while(qh<qt){
            int idx2 = queue[qh++];
            pg_node *n = order->items[idx2];
            for(size_t pi=0; pi<n->nparents; pi++){
                pg_node *par = n->parents[pi];
                int p_idx = find_order_idx(order, par);
                if(p_idx<0) continue;
                if(in_suffix[p_idx]) continue;
                pg_node *pn = order->items[p_idx];
                if(pn->ag_op==PG_AG_OP_NONE) continue;
                if(!ag_op_is_jit_elementwise(pn->ag_op)) continue;
                if (pn->value->ndim != loop_ndim || pn->value->numel != loop_numel ||
                    !pg_shape_equal(pn->value->ndim, pn->value->shape, loop_ndim, loop_shape))
                    continue;
                in_suffix[p_idx]=true;
                queue[qt++]=p_idx;
            }
        }
    }
    free(queue);
    *out_handled = in_suffix;
    return true;
}

bool pg_backward_jit(pg_node *loss) {
    if (!loss || !loss->requires_grad) return false;
    node_vec order = {0};
    topo_sort(loss, &order);
    bool ok = try_jit_backward_internal(loss, &order);
    for (size_t i = 0; i < order.n; i++) order.items[i]->mark = 0;
    free(order.items);
    return ok;
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

    bool *in_suffix = NULL;
    bool did_jit = false;
    if (g_ag_jit_enabled) {
        if (try_jit_suffix(loss, &order, &in_suffix)) {
            did_jit = true;
            g_ag_last_was_jit = true;
            g_ag_jit_hits++;
        }
    }
    if (did_jit) {
        // suffix JIT succeeded, handle remaining nodes eagerly
        for (size_t i = order.n; i-- > 0;) {
            pg_node *n = order.items[i];
            size_t idx = 0;
            for (size_t k=0;k<order.n;k++) if(order.items[k]==n) {idx=k; break;}
            if (in_suffix && in_suffix[idx]) continue;
            if (n->backward && n->grad)
                n->backward(n);
        }
        free(in_suffix);
    } else {
        if (g_ag_jit_enabled) {
            g_ag_last_was_jit = false;
            g_ag_jit_fallbacks++;
        }
        if (in_suffix) free(in_suffix);
        ensure_grad(loss);
        pg_tensor_fill(loss->grad, 1.0f);
        for (size_t i = order.n; i-- > 0;) {
            pg_node *n = order.items[i];
            if (n->backward && n->grad)
                n->backward(n);
        }
    }

    for (size_t i = 0; i < order.n; i++)
        order.items[i]->mark = 0;

    free(order.items);
}
