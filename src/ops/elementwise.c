#include "elementwise.h"

#include "common.h"
#include "../backend/backend.h"
#include "../thread/pool.h"
#include <assert.h>
#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float fadd(float x, float y) { return x + y; }
static float fsub(float x, float y) { return x - y; }
static float fmul(float x, float y) { return x * y; }
static float fdiv(float x, float y) { return x / y; }
static float fpow(float x, float y) { return powf(x, y); }

static inline void avx_bin_add(const float *a, const float *b, float *o, size_t n){
    size_t i=0;
    for(; i+8<=n; i+=8){ __m256 va=_mm256_loadu_ps(a+i); __m256 vb=_mm256_loadu_ps(b+i); __m256 vc=_mm256_add_ps(va,vb); _mm256_storeu_ps(o+i, vc); }
    for(; i<n; i++) o[i]=a[i]+b[i];
}
static inline void avx_bin_sub(const float *a, const float *b, float *o, size_t n){
    size_t i=0;
    for(; i+8<=n; i+=8){ __m256 va=_mm256_loadu_ps(a+i); __m256 vb=_mm256_loadu_ps(b+i); __m256 vc=_mm256_sub_ps(va,vb); _mm256_storeu_ps(o+i, vc); }
    for(; i<n; i++) o[i]=a[i]-b[i];
}
static inline void avx_bin_mul(const float *a, const float *b, float *o, size_t n){
    size_t i=0;
    for(; i+8<=n; i+=8){ __m256 va=_mm256_loadu_ps(a+i); __m256 vb=_mm256_loadu_ps(b+i); __m256 vc=_mm256_mul_ps(va,vb); _mm256_storeu_ps(o+i, vc); }
    for(; i<n; i++) o[i]=a[i]*b[i];
}
static inline void avx_bin_div(const float *a, const float *b, float *o, size_t n){
    size_t i=0;
    for(; i+8<=n; i+=8){ __m256 va=_mm256_loadu_ps(a+i); __m256 vb=_mm256_loadu_ps(b+i); __m256 vc=_mm256_div_ps(va,vb); _mm256_storeu_ps(o+i, vc); }
    for(; i<n; i++) o[i]=a[i]/b[i];
}
static inline void avx_scalar_add(const float *a, float bv, float *o, size_t n){
    __m256 vb=_mm256_set1_ps(bv);
    size_t i=0;
    for(; i+8<=n; i+=8){ __m256 va=_mm256_loadu_ps(a+i); __m256 vc=_mm256_add_ps(va,vb); _mm256_storeu_ps(o+i, vc); }
    for(; i<n; i++) o[i]=a[i]+bv;
}

// parallel helpers for contiguous binary ops
typedef struct { const float *a; const float *b; float *out; } bin_par_t;
static void par_add(void *ctx, size_t s, size_t e){ bin_par_t *p=ctx; avx_bin_add(p->a+s, p->b+s, p->out+s, e-s); }
static void par_sub(void *ctx, size_t s, size_t e){ bin_par_t *p=ctx; avx_bin_sub(p->a+s, p->b+s, p->out+s, e-s); }
static void par_mul(void *ctx, size_t s, size_t e){ bin_par_t *p=ctx; avx_bin_mul(p->a+s, p->b+s, p->out+s, e-s); }
static void par_div(void *ctx, size_t s, size_t e){ bin_par_t *p=ctx; avx_bin_div(p->a+s, p->b+s, p->out+s, e-s); }
static void par_pow(void *ctx, size_t s, size_t e){ bin_par_t *p=ctx; const float *a=p->a; const float *b=p->b; float *o=p->out; for(size_t i=s;i<e;i++) o[i]=powf(a[i],b[i]); }

typedef struct { float av; const float *b; float *out; float (*f)(float,float); } scalar_par_t;
static void par_scalar_a(void *ctx, size_t s, size_t e){ scalar_par_t *p=ctx; float av=p->av; const float *b=p->b; float *o=p->out; float (*f)(float,float)=p->f; for(size_t i=s;i<e;i++) o[i]=f(av,b[i]); }
static void par_scalar_b(void *ctx, size_t s, size_t e){ scalar_par_t *p=ctx; float bv=p->av; const float *a=p->b; float *o=p->out; float (*f)(float,float)=p->f; for(size_t i=s;i<e;i++) o[i]=f(a[i],bv); }

typedef struct { float av; const float *b; float *out; } scalar_add_t;
static void par_scalar_add_a(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float av=p->av; const float *b=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=av+b[i]; }
static void par_scalar_sub_a(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float av=p->av; const float *b=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=av-b[i]; }
static void par_scalar_mul_a(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float av=p->av; const float *b=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=av*b[i]; }
static void par_scalar_div_a(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float av=p->av; const float *b=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=av/b[i]; }
static void par_scalar_add_b(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float bv=p->av; const float *a=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=a[i]+bv; }
static void par_scalar_sub_b(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float bv=p->av; const float *a=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=a[i]-bv; }
static void par_scalar_mul_b(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float bv=p->av; const float *a=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=a[i]*bv; }
static void par_scalar_div_b(void *ctx, size_t s, size_t e){ scalar_add_t *p=ctx; float bv=p->av; const float *a=p->b; float *o=p->out;
#pragma GCC ivdep
for(size_t i=s;i<e;i++) o[i]=a[i]/bv; }

typedef struct { const float *pa; const float *pb; float *po; size_t N; float (*f)(float,float); } row_bcast_t;
static void row_bcast_fn(void *ctx, size_t s, size_t e){ row_bcast_t *p=ctx; for(size_t i=s;i<e;i++){ const float *par=p->pa+i*p->N; float *por=p->po+i*p->N; const float *pb=p->pb; float (*ff)(float,float)=p->f;
#pragma GCC ivdep
for(size_t j=0;j<p->N;j++) por[j]=ff(par[j],pb[j]); } }
typedef struct { const float *pb; const float *pa; float *po; size_t N; float (*f)(float,float); } row_bcast2_t;
static void row_bcast2_fn(void *ctx, size_t s, size_t e){ row_bcast2_t *p=ctx; for(size_t i=s;i<e;i++){ const float *pbr=p->pb+i*p->N; float *por=p->po+i*p->N; const float *pa=p->pa; float (*ff)(float,float)=p->f;
#pragma GCC ivdep
for(size_t j=0;j<p->N;j++) por[j]=ff(pa[j],pbr[j]); } }

// helper to dispatch vectorized parallel for same-shape contiguous
static inline void dispatch_contiguous(const float *pa, const float *pb, float *po, size_t n, float (*f)(float,float)) {
    if (n < 262144) {
        if (f==fadd) { avx_bin_add(pa,pb,po,n); }
        else if (f==fsub) { avx_bin_sub(pa,pb,po,n); }
        else if (f==fmul) { avx_bin_mul(pa,pb,po,n); }
        else if (f==fdiv) { avx_bin_div(pa,pb,po,n); }
        else if (f==fpow) { for (size_t i=0;i<n;i++) po[i]=powf(pa[i],pb[i]); }
        else { for (size_t i=0;i<n;i++) po[i]=f(pa[i],pb[i]); }
        return;
    }
    bin_par_t ctx={pa,pb,po};
    if (f==fadd) pg_parallel_for(n, 65536, par_add, &ctx);
    else if (f==fsub) pg_parallel_for(n, 65536, par_sub, &ctx);
    else if (f==fmul) pg_parallel_for(n, 65536, par_mul, &ctx);
    else if (f==fdiv) pg_parallel_for(n, 65536, par_div, &ctx);
    else if (f==fpow) pg_parallel_for(n, 65536, par_pow, &ctx);
    else {
        #pragma GCC ivdep
        for (size_t i=0;i<n;i++) po[i]=f(pa[i],pb[i]);
    }
}

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

    pg_tensor *out = pg_tensor_empty(ndim, shape);
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

    pg_status st = pg_op_bin(dc.ptr, da.ptr, db.ptr, out->numel, bin_op, &kargs);
    if (st != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&db); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
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

static inline bool is_contiguous(const pg_tensor *t) {
    size_t acc = 1;
    for (size_t i = t->ndim; i-- > 0;) {
        if (t->stride[i] != acc) return false;
        acc *= t->shape[i];
    }
    return true;
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

    pg_tensor *out = pg_tensor_empty(ndim, shape);
    if (!out)
        return NULL;

    // Fast path: same shape, both contiguous -> single linear loop (vectorizable)
    if (a->numel == out->numel && b->numel == out->numel &&
        a->ndim == ndim && b->ndim == ndim &&
        memcmp(a->shape, shape, ndim * sizeof(size_t)) == 0 &&
        memcmp(b->shape, shape, ndim * sizeof(size_t)) == 0 &&
        is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
        const float *pa = a->data;
        const float *pb = b->data;
        float *po = out->data;
        size_t n = out->numel;
        dispatch_contiguous(pa,pb,po,n,f);
        return out;
    }
    // Fast path: scalar broadcast (numel==1)
    if (a->numel == 1 && is_contiguous(a)) {
        float av = a->data[0];
        const float *pb = b->data;
        float *po = out->data;
        if (b->numel == out->numel && is_contiguous(b)) {
            size_t n = out->numel;
            if (n < 262144) {
                if (f==fadd) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=av+pb[i];
                } else if (f==fsub) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=av-pb[i];
                } else if (f==fmul) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=av*pb[i];
                } else if (f==fdiv) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=av/pb[i];
                } else for (size_t i=0;i<n;i++) po[i]=f(av,pb[i]);
            } else {
                if (f==fadd) { scalar_add_t ctx={av,pb,po}; pg_parallel_for(n,65536,par_scalar_add_a,&ctx); }
                else if (f==fsub) { scalar_add_t ctx={av,pb,po}; pg_parallel_for(n,65536,par_scalar_sub_a,&ctx); }
                else if (f==fmul) { scalar_add_t ctx={av,pb,po}; pg_parallel_for(n,65536,par_scalar_mul_a,&ctx); }
                else if (f==fdiv) { scalar_add_t ctx={av,pb,po}; pg_parallel_for(n,65536,par_scalar_div_a,&ctx); }
                else { scalar_par_t ctx={av,pb,po,f}; pg_parallel_for(n,65536,par_scalar_a,&ctx); }
            }
            return out;
        }
    }
    if (b->numel == 1 && is_contiguous(b)) {
        float bv = b->data[0];
        const float *pa = a->data;
        float *po = out->data;
        if (a->numel == out->numel && is_contiguous(a)) {
            size_t n = out->numel;
            if (n < 262144) {
                if (f==fadd) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=pa[i]+bv;
                } else if (f==fsub) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=pa[i]-bv;
                } else if (f==fmul) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=pa[i]*bv;
                } else if (f==fdiv) {
#pragma GCC ivdep
                    for(size_t i=0;i<n;i++) po[i]=pa[i]/bv;
                } else for (size_t i=0;i<n;i++) po[i]=f(pa[i],bv);
            } else {
                if (f==fadd) { scalar_add_t ctx={bv,pa,po}; pg_parallel_for(n,65536,par_scalar_add_b,&ctx); }
                else if (f==fsub) { scalar_add_t ctx={bv,pa,po}; pg_parallel_for(n,65536,par_scalar_sub_b,&ctx); }
                else if (f==fmul) { scalar_add_t ctx={bv,pa,po}; pg_parallel_for(n,65536,par_scalar_mul_b,&ctx); }
                else if (f==fdiv) { scalar_add_t ctx={bv,pa,po}; pg_parallel_for(n,65536,par_scalar_div_b,&ctx); }
                else { scalar_par_t ctx={bv,pa,po,f}; pg_parallel_for(n,65536,par_scalar_b,&ctx); }
            }
            return out;
        }
    }
    // Fast path: row broadcast [M,N] + [N] (common bias add) and similar 2D case
    if (ndim == 2 && shape[1] > 1) {
        if (a->ndim == 2 && b->ndim == 1 &&
            a->shape[0] == shape[0] && a->shape[1] == shape[1] &&
            b->shape[0] == shape[1] &&
            is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
            size_t M = shape[0], N = shape[1];
            const float *pa = a->data;
            const float *pb = b->data;
            float *po = out->data;
            if (M * N < 8192 || M < 4) {
                for (size_t i = 0; i < M; i++) {
                    const float *par = pa + i * N;
                    float *por = po + i * N;
                    #pragma GCC ivdep
                    for (size_t j = 0; j < N; j++) por[j] = f(par[j], pb[j]);
                }
            } else {
                row_bcast_t ctx={pa,pb,po,N,f};
                pg_parallel_for(M, 4, row_bcast_fn, &ctx);
            }
            return out;
        }
        if (b->ndim == 2 && a->ndim == 1 &&
            b->shape[0] == shape[0] && b->shape[1] == shape[1] &&
            a->shape[0] == shape[1] &&
            is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
            size_t M = shape[0], N = shape[1];
            const float *pb = b->data;
            const float *pa = a->data;
            float *po = out->data;
            if (M * N < 8192 || M < 4) {
                for (size_t i = 0; i < M; i++) {
                    const float *pbr = pb + i * N;
                    float *por = po + i * N;
                    #pragma GCC ivdep
                    for (size_t j = 0; j < N; j++) por[j] = f(pa[j], pbr[j]);
                }
            } else {
                row_bcast2_t ctx={pb,pa,po,N,f};
                pg_parallel_for(M, 4, row_bcast2_fn, &ctx);
            }
            return out;
        }
    }
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
    pg_tensor *out = pg_tensor_empty(a->ndim, a->shape);
    if (!out)
        return NULL;
    size_t bytes = a->numel * sizeof(float);
    pg_dev_buf da = pg_dev_buf_new(bytes ? bytes : 1);
    pg_dev_buf dc = pg_dev_buf_new(bytes ? bytes : 1);
    if (!da.ptr || !dc.ptr) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_h2d(da.ptr, a->data, bytes) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_status st = pg_op_map(dc.ptr, da.ptr, a->numel, map_op);
    if (st != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_dev_sync() != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    if (pg_copy_d2h(out->data, dc.ptr, bytes) != PG_OK) {
        pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
        pg_tensor_free(out);
        return NULL;
    }
    pg_dev_buf_free(&da); pg_dev_buf_free(&dc);
    return out;
}

typedef struct { float *d; float (*f)(float); } map_par_t;
static void map_par_fn(void *ctx, size_t s, size_t e){ map_par_t *p=ctx; float *d=p->d; float (*ff)(float)=p->f; for(size_t i=s;i<e;i++) d[i]=ff(d[i]); }

static pg_tensor *map1(const pg_tensor *a, float (*f)(float))
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    float *restrict pr = r->data;
    size_t n = r->numel;
    if (n < 65536) {
        #pragma GCC ivdep
        for (size_t i = 0; i < n; i++) pr[i] = f(pr[i]);
    } else {
        map_par_t ctx={pr,f};
        pg_parallel_for(n, 65536, map_par_fn, &ctx);
    }
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

typedef struct { float *d; float lo; float hi; } clamp_par_t;
static void clamp_par_fn(void *ctx, size_t s, size_t e){ clamp_par_t *p=ctx; float *d=p->d; float lo=p->lo, hi=p->hi; for(size_t i=s;i<e;i++){ float v=d[i]; d[i]= v<lo?lo:(v>hi?hi:v); } }

pg_tensor *pg_clamp(const pg_tensor *a, float lo, float hi)
{
    assert(a && a->data);
    pg_tensor *r = pg_tensor_clone(a);
    if (!r)
        return NULL;
    float *restrict pr = r->data;
    size_t n = r->numel;
    if (n < 65536) {
        #pragma GCC ivdep
        for (size_t i = 0; i < n; i++) {
            float v = pr[i];
            pr[i] = v < lo ? lo : (v > hi ? hi : v);
        }
    } else {
        clamp_par_t ctx={pr,lo,hi};
        pg_parallel_for(n, 65536, clamp_par_fn, &ctx);
    }
    return r;
}
