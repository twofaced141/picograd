#include "activations.h"

#include "common.h"
#include "simd.h"
#include "../backend/backend.h"
#include "../thread/pool.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *d; float (*f)(float); } act_map_par_t;
static void act_map_par_fn(void *ctx, size_t s, size_t e){
    act_map_par_t *p = ctx;
    float *d = p->d;
    float (*ff)(float) = p->f;
    for (size_t i = s; i < e; i++)
        d[i] = ff(d[i]);
}

typedef struct { const float *src; float *dst; size_t len; bool log; } sm1_par_t;
static void sm1_par_fn(void *ctx, size_t s, size_t e){
    sm1_par_t *p = ctx;
    for (size_t o = s; o < e; o++) {
        const float *srow = p->src + o * p->len;
        float *drow = p->dst + o * p->len;
        float m = srow[0];
        for (size_t j = 1; j < p->len; j++)
            if (srow[j] > m)
                m = srow[j];
        float s = 0.0f;
        for (size_t j = 0; j < p->len; j++)
            s += expf(srow[j] - m);
        if (p->log) {
            float ls = logf(s);
            for (size_t j = 0; j < p->len; j++)
                drow[j] = srow[j] - m - ls;
        } else {
            float inv = 1.0f / s;
            for (size_t j = 0; j < p->len; j++)
                drow[j] = expf(srow[j] - m) * inv;
        }
    }
}

static pg_tensor *map_inplace(const pg_tensor *a, float (*f)(float))
{
    if (!a || !a->data || !f) return NULL;
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    float *restrict pr = r->data;
    size_t n = r->numel;
    if (n < 65536) {
        #pragma GCC ivdep
        for (size_t i = 0; i < n; i++) pr[i] = f(pr[i]);
    } else {
        act_map_par_t ctx={pr,f};
        pg_parallel_for(n, 65536, act_map_par_fn, &ctx);
    }
    return r;
}

static pg_tensor *try_map_gpu_act(const pg_tensor *a, int op)
{
    if (pg_get_device() == PG_DEV_CPU)
        return NULL;
    if (a->numel > UINT_MAX)
        return NULL;
    pg_tensor *out = pg_tensor_empty(a->ndim, a->shape);
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
    pg_status st = pg_op_map(dc, da, a->numel, op);
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

static float fsigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float fgelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); }

static inline void relu_simd(float *d, size_t n) {
    simd_relu(d, n);
}
static inline void leaky_simd(float *d, size_t n, float alpha) {
    simd_leaky_relu(d, n, alpha);
}
// compat aliases for previous avx names
#define relu_avx relu_simd
#define leaky_avx leaky_simd
typedef struct { float *d; } relu_par_t;
static void relu_par_fn(void *ctx, size_t s, size_t e) {
    relu_par_t *p = (relu_par_t*)ctx;
    relu_simd(p->d + s, e - s);
}
typedef struct { float *d; float alpha; } leaky_par2_t;
static void leaky_par2_fn(void *ctx, size_t s, size_t e) {
    leaky_par2_t *p = (leaky_par2_t*)ctx;
    leaky_simd(p->d + s, e - s, p->alpha);
}

pg_tensor *pg_relu(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_RELU);
    if (g) return g;
    pg_tensor *r = pg_tensor_clone(a);
    if(!r) return NULL;
    float *pr = r->data;
    size_t n = r->numel;
    if (n < 65536) {
        if (n >= 32) simd_relu(pr, n);
        else {
            for (size_t i = 0; i < n; i++) pr[i] = pr[i] > 0 ? pr[i] : 0;
        }
    } else {
        relu_par_t ctx={pr};
        pg_parallel_for(n, 65536, relu_par_fn, &ctx);
    }
    return r;
}

pg_tensor *pg_sigmoid(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_SIGMOID);
    if (g) return g;
    return map_inplace(a, fsigmoid);
}

pg_tensor *pg_tanh(const pg_tensor *a)
{
    pg_tensor *g = try_map_gpu_act(a, PG_MAP_TANH);
    if (g) return g;
    return map_inplace(a, tanhf);
}

pg_tensor *pg_gelu(const pg_tensor *a)
{
    return map_inplace(a, fgelu);
}

pg_tensor *pg_leaky_relu(const pg_tensor *a, float alpha)
{
    if (!a || !a->data) return NULL;
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    float *restrict pr = r->data;
    size_t n = r->numel;
    if (n < 65536) {
        if(n >= 32) simd_leaky_relu(pr, n, alpha);
        else for (size_t i = 0; i < n; i++) { float v = pr[i]; pr[i] = v > 0.0f ? v : alpha * v; }
    } else {
        leaky_par2_t ctx={pr,alpha};
        pg_parallel_for(n, 65536, leaky_par2_fn, &ctx);
    }
    return r;
}

static pg_tensor *softmax_impl(const pg_tensor *t, size_t axis, bool log)
{
    if (!t || !t->data || axis >= t->ndim) return NULL;

    size_t outer, len, inner;
    pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);

    pg_tensor *r = pg_tensor_clone(t);
    if (!r)
        return NULL;

    const float *src = t->data;
    float *dst = r->data;

    // Fast path for inner==1 (last axis) -> contiguous rows, vectorizable and cache-friendly
    if (inner == 1) {
        if (outer < 32 || outer * len < 16384) {
            for (size_t o = 0; o < outer; o++) {
                const float *srow = src + o * len;
                float *drow = dst + o * len;
                float m = srow[0];
                for (size_t j = 1; j < len; j++) if (srow[j] > m) m = srow[j];
                float s = 0.0f;
                for (size_t j = 0; j < len; j++) s += expf(srow[j] - m);
                if (log) {
                    float ls = logf(s);
                    for (size_t j = 0; j < len; j++) drow[j] = srow[j] - m - ls;
                } else {
                    float inv = 1.0f / s;
                    for (size_t j = 0; j < len; j++) drow[j] = expf(srow[j] - m) * inv;
                }
            }
        } else {
            sm1_par_t ctx={src,dst,len,log};
            pg_parallel_for(outer, 32, sm1_par_fn, &ctx);
        }
        return r;
    }

    // General case: per outer, process len x inner tile with contiguous inner loops
    for (size_t o = 0; o < outer; o++) {
        const float *base_src = src + o * len * inner;
        float *base_dst = dst + o * len * inner;
        // per-inner max/sum: stack buffer up to 1024, else heap
        float *max_buf = NULL;
        float *sum_buf = NULL;
        float max_stack[1024];
        float sum_stack[1024];
        bool use_heap = inner > 1024;
        if (use_heap) {
            max_buf = malloc(inner * sizeof(float));
            sum_buf = malloc(inner * sizeof(float));
            if (!max_buf || !sum_buf) goto fallback;
        } else {
            max_buf = max_stack;
            sum_buf = sum_stack;
        }
        for (size_t ii = 0; ii < inner; ii++) max_buf[ii] = base_src[ii];
        for (size_t v = 1; v < len; v++) {
            const float *row = base_src + v * inner;
            #pragma GCC ivdep
            for (size_t ii = 0; ii < inner; ii++) if (row[ii] > max_buf[ii]) max_buf[ii] = row[ii];
        }
        for (size_t ii = 0; ii < inner; ii++) sum_buf[ii] = 0.0f;
        for (size_t v = 0; v < len; v++) {
            const float *row = base_src + v * inner;
            #pragma GCC ivdep
            for (size_t ii = 0; ii < inner; ii++) sum_buf[ii] += expf(row[ii] - max_buf[ii]);
        }
        if (log) {
            for (size_t ii = 0; ii < inner; ii++) sum_buf[ii] = logf(sum_buf[ii]);
            for (size_t v = 0; v < len; v++) {
                const float *row = base_src + v * inner;
                float *drow = base_dst + v * inner;
                #pragma GCC ivdep
                for (size_t ii = 0; ii < inner; ii++) drow[ii] = row[ii] - max_buf[ii] - sum_buf[ii];
            }
        } else {
            for (size_t ii = 0; ii < inner; ii++) sum_buf[ii] = 1.0f / sum_buf[ii];
            for (size_t v = 0; v < len; v++) {
                const float *row = base_src + v * inner;
                float *drow = base_dst + v * inner;
                #pragma GCC ivdep
                for (size_t ii = 0; ii < inner; ii++) drow[ii] = expf(row[ii] - max_buf[ii]) * sum_buf[ii];
            }
        }
        if (use_heap) { free(max_buf); free(sum_buf); }
        continue;
fallback:
        // fallback: per-inner strided (slower path used when heap alloc fails)
        for (size_t i = 0; i < inner; i++) {
            const float *scol = base_src + i;
            float *dcol = base_dst + i;
            float m = -INFINITY;
            for (size_t j = 0; j < len; j++) {
                float v = scol[j * inner];
                if (v > m) m = v;
            }
            float s = 0.0f;
            for (size_t j = 0; j < len; j++) s += expf(scol[j * inner] - m);
            if (log) {
                float ls = logf(s);
                for (size_t j = 0; j < len; j++) dcol[j * inner] = scol[j * inner] - m - ls;
            } else {
                float inv = 1.0f / s;
                for (size_t j = 0; j < len; j++) dcol[j * inner] = expf(scol[j * inner] - m) * inv;
            }
        }
        if (use_heap) { free(max_buf); free(sum_buf); }
    }
    return r;
}

pg_tensor *pg_softmax(const pg_tensor *t, size_t axis)
{
    if (pg_get_device() != PG_DEV_CPU && t->numel <= UINT_MAX) {
        size_t outer, len, inner;
        pg_axis_split(t->ndim, t->shape, axis, &outer, &len, &inner);
        if (outer <= UINT_MAX && len <= UINT_MAX && inner <= UINT_MAX) {
            pg_tensor *out = pg_tensor_empty(t->ndim, t->shape);
            if (out) {
                size_t bytes = t->numel * sizeof(float);
                float *da = pg_dev_malloc(bytes ? bytes : 1);
                float *dc = pg_dev_malloc(bytes ? bytes : 1);
                if (da && dc &&
                    pg_copy_h2d(da, t->data, bytes) == PG_OK &&
                    pg_op_softmax(dc, da, outer, len, inner) == PG_OK &&
                    pg_dev_sync() == PG_OK &&
                    pg_copy_d2h(out->data, dc, bytes) == PG_OK) {
                    pg_dev_free(da);
                    pg_dev_free(dc);
                    return out;
                }
                if (da) pg_dev_free(da);
                if (dc) pg_dev_free(dc);
                pg_tensor_free(out);
            }
        }
    }
    return softmax_impl(t, axis, false);
}

pg_tensor *pg_log_softmax(const pg_tensor *t, size_t axis)
{
    return softmax_impl(t, axis, true);
}
