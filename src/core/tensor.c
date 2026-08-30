#include "tensor.h"
#include "convert.h"

#include "../thread/pool.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *d; float v; } fill_par_t;
static void fill_par_fn(void *ctx, size_t s, size_t e){ fill_par_t *p=ctx; float *d=p->d; float v=p->v; for(size_t i=s;i<e;i++) d[i]=v; }

/* ---------- aligned pool ---------- */
#define PG_POOL_MAX_COUNT 64
#define PG_POOL_MAX_BYTES (64*1024*1024)
#define PG_TENSOR_ALIGN 64

typedef struct pool_entry {
    size_t nbytes;
    void *ptr;
    struct pool_entry *next;
} pool_entry_t;

static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pool_entry_t *g_pool_head = NULL;
static size_t g_pool_count = 0;
static size_t g_pool_bytes = 0;

static void *pg_aligned_alloc(size_t nbytes) {
    void *p = NULL;
    size_t aligned_nbytes = (nbytes + PG_TENSOR_ALIGN - 1) & ~(PG_TENSOR_ALIGN - 1);
    if (aligned_nbytes < nbytes) aligned_nbytes = nbytes;
    if (posix_memalign(&p, PG_TENSOR_ALIGN, aligned_nbytes) != 0) return NULL;
    return p;
}

static void *pool_acquire(size_t nbytes) {
    pthread_mutex_lock(&g_pool_mu);
    pool_entry_t **pp = &g_pool_head;
    pool_entry_t *cur = g_pool_head;
    pool_entry_t *best = NULL;
    pool_entry_t **best_pp = NULL;
    size_t best_sz = SIZE_MAX;
    while (cur) {
        if (cur->nbytes == nbytes) { best = cur; best_pp = pp; break; }
        if (cur->nbytes >= nbytes && cur->nbytes < best_sz) { best = cur; best_pp = pp; best_sz = cur->nbytes; }
        pp = &cur->next;
        cur = cur->next;
    }
    if (best) {
        *best_pp = best->next;
        void *ptr = best->ptr;
        size_t sz = best->nbytes;
        free(best);
        g_pool_count--;
        g_pool_bytes -= sz;
        pthread_mutex_unlock(&g_pool_mu);
        return ptr;
    }
    pthread_mutex_unlock(&g_pool_mu);
    return NULL;
}

static void pool_release(void *ptr, size_t nbytes) {
    if (!ptr) return;
    pthread_mutex_lock(&g_pool_mu);
    if (g_pool_count >= PG_POOL_MAX_COUNT || g_pool_bytes + nbytes > PG_POOL_MAX_BYTES) {
        pthread_mutex_unlock(&g_pool_mu);
        free(ptr);
        return;
    }
    pool_entry_t *e = (pool_entry_t*)malloc(sizeof(*e));
    if (!e) { pthread_mutex_unlock(&g_pool_mu); free(ptr); return; }
    e->ptr = ptr;
    e->nbytes = nbytes;
    e->next = g_pool_head;
    g_pool_head = e;
    g_pool_count++;
    g_pool_bytes += nbytes;
    pthread_mutex_unlock(&g_pool_mu);
}

void pg_tensor_pool_clear(void) {
    pthread_mutex_lock(&g_pool_mu);
    pool_entry_t *cur = g_pool_head;
    while (cur) { pool_entry_t *n = cur->next; free(cur->ptr); free(cur); cur = n; }
    g_pool_head = NULL; g_pool_count = 0; g_pool_bytes = 0;
    pthread_mutex_unlock(&g_pool_mu);
}
size_t pg_tensor_pool_size(void) { size_t c; pthread_mutex_lock(&g_pool_mu); c=g_pool_count; pthread_mutex_unlock(&g_pool_mu); return c; }
size_t pg_tensor_pool_bytes(void) { size_t b; pthread_mutex_lock(&g_pool_mu); b=g_pool_bytes; pthread_mutex_unlock(&g_pool_mu); return b; }

static pg_data_ref *data_ref_create(void *ptr, size_t nbytes) {
    pg_data_ref *r = (pg_data_ref*)malloc(sizeof(*r));
    if (!r) return NULL;
    r->ptr = ptr;
    r->nbytes = nbytes;
    r->refs = 1;
    return r;
}
static void data_ref_retain(pg_data_ref *r) {
    if (r) __sync_fetch_and_add(&r->refs, 1);
}
static void data_ref_release(pg_data_ref *r) {
    if (!r) return;
    if (__sync_sub_and_fetch(&r->refs, 1) == 0) {
        if (r->ptr) {
            pthread_mutex_lock(&g_pool_mu);
            if (g_pool_count >= PG_POOL_MAX_COUNT || g_pool_bytes + r->nbytes > PG_POOL_MAX_BYTES) {
                pthread_mutex_unlock(&g_pool_mu);
                free(r->ptr);
            } else {
                pool_entry_t *e = (pool_entry_t*)malloc(sizeof(*e));
                if (!e) { pthread_mutex_unlock(&g_pool_mu); free(r->ptr); free(r); return; }
                e->ptr = r->ptr;
                e->nbytes = r->nbytes;
                e->next = g_pool_head;
                g_pool_head = e;
                g_pool_count++;
                g_pool_bytes += r->nbytes;
                pthread_mutex_unlock(&g_pool_mu);
            }
        }
        free(r);
    }
}

static bool valid_shape(size_t ndim, const size_t *shape)
{
    if (ndim == 0 || ndim > PG_MAX_NDIM || shape == NULL)
        return false;
    for (size_t i = 0; i < ndim; i++)
        if (shape[i] == 0)
            return false;
    return true;
}

static void compute_strides(size_t ndim, const size_t *shape, size_t *stride)
{
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0;) {
        stride[i] = acc;
        acc *= shape[i];
    }
}

static pg_tensor *tensor_alloc_dtype(pg_dtype dtype, size_t ndim, const size_t *shape, bool zero) {
    if (!valid_shape(ndim, shape)) return NULL;
    size_t elem_size = pg_dtype_size(dtype);
    if (elem_size == 0) return NULL;
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (numel > SIZE_MAX / shape[i]) return NULL;
        numel *= shape[i];
    }
    if (numel > SIZE_MAX / elem_size) return NULL;
    pg_tensor *t = malloc(sizeof(*t));
    if (!t) return NULL;
    size_t nbytes = numel * elem_size;
    void *data = pool_acquire(nbytes);
    if (!data) {
        data = pg_aligned_alloc(nbytes);
        if (!data) { free(t); return NULL; }
    }
    if (zero) memset(data, 0, nbytes);
    pg_data_ref *ref = data_ref_create(data, nbytes);
    if (!ref) { pool_release(data, nbytes); free(t); return NULL; }
    t->data_raw = data;
    t->data_ref = ref;
    t->is_view = false;
    t->view_parent = NULL;
    t->dtype = dtype;
    t->elem_size = elem_size;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, t->shape, t->stride);
    t->numel = numel;
    return t;
}

pg_tensor *pg_tensor_new_dtype(pg_dtype dtype, size_t ndim, const size_t *shape){
    return tensor_alloc_dtype(dtype, ndim, shape, true);
}
pg_tensor *pg_tensor_empty_dtype(pg_dtype dtype, size_t ndim, const size_t *shape){
    return tensor_alloc_dtype(dtype, ndim, shape, false);
}
pg_tensor *pg_tensor_zeros_dtype(pg_dtype dtype, size_t ndim, const size_t *shape){
    return tensor_alloc_dtype(dtype, ndim, shape, true);
}
pg_tensor *pg_tensor_ones_dtype(pg_dtype dtype, size_t ndim, const size_t *shape){
    pg_tensor *t = tensor_alloc_dtype(dtype, ndim, shape, false);
    if (!t) return NULL;
    pg_tensor_fill(t, 1.0f);
    return t;
}
pg_tensor *pg_tensor_full_dtype(pg_dtype dtype, size_t ndim, const size_t *shape, float value){
    pg_tensor *t = tensor_alloc_dtype(dtype, ndim, shape, false);
    if (!t) return NULL;
    pg_tensor_fill(t, value);
    return t;
}
pg_tensor *pg_tensor_from_data_dtype(pg_dtype dtype, size_t ndim, const size_t *shape, const void *data){
    if (!data) return NULL;
    pg_tensor *t = pg_tensor_empty_dtype(dtype, ndim, shape);
    if (!t) return NULL;
    memcpy(t->data_raw, data, t->numel * t->elem_size);
    return t;
}

pg_tensor *pg_tensor_new(size_t ndim, const size_t *shape)
{
    return pg_tensor_new_dtype(PG_DTYPE_F32, ndim, shape);
}

pg_tensor *pg_tensor_empty(size_t ndim, const size_t *shape)
{
    return pg_tensor_empty_dtype(PG_DTYPE_F32, ndim, shape);
}

void pg_tensor_free(pg_tensor *t)
{
    if (!t)
        return;
    if (t->data_ref) data_ref_release(t->data_ref);
    free(t);
}

pg_tensor *pg_tensor_full(size_t ndim, const size_t *shape, float value)
{
    return pg_tensor_full_dtype(PG_DTYPE_F32, ndim, shape, value);
}

pg_tensor *pg_tensor_zeros(size_t ndim, const size_t *shape)
{
    return pg_tensor_new_dtype(PG_DTYPE_F32, ndim, shape);
}

pg_tensor *pg_tensor_ones(size_t ndim, const size_t *shape)
{
    return pg_tensor_full_dtype(PG_DTYPE_F32, ndim, shape, 1.0f);
}

pg_tensor *pg_tensor_from_data(size_t ndim, const size_t *shape, const float *data)
{
    return pg_tensor_from_data_dtype(PG_DTYPE_F32, ndim, shape, data);
}

pg_tensor *pg_tensor_arange(float start, float stop, float step)
{
    if (step <= 0.0f || stop <= start)
        return NULL;
    size_t n = (size_t)ceilf((stop - start) / step);
    pg_tensor *t = pg_tensor_empty(1, &n);
    if (!t)
        return NULL;
    for (size_t i = 0; i < n; i++)
        t->data[i] = start + step * (float)i;
    return t;
}

static unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;
static pthread_mutex_t g_rng_mu = PTHREAD_MUTEX_INITIALIZER;

void pg_seed(unsigned long long seed)
{
    pthread_mutex_lock(&g_rng_mu);
    rng_state = seed != 0 ? seed : 0x9E3779B97F4A7C15ULL;
    pthread_mutex_unlock(&g_rng_mu);
}

static unsigned long long rng_next(void)
{
    pthread_mutex_lock(&g_rng_mu);
    rng_state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = rng_state;
    pthread_mutex_unlock(&g_rng_mu);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rng_u01(void)
{
    return (float)(rng_next() >> 40) * (1.0f / 16777216.0f);
}

pg_tensor *pg_tensor_uniform(size_t ndim, const size_t *shape, float low, float high)
{
    pg_tensor *t = pg_tensor_empty(ndim, shape);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] = low + rng_u01() * (high - low);
    return t;
}

pg_tensor *pg_tensor_normal(size_t ndim, const size_t *shape, float mean, float stddev)
{
    pg_tensor *t = pg_tensor_empty(ndim, shape);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++) {
        float u1 = rng_u01();
        float u2 = rng_u01();
        if (u1 < 1e-12f)
            u1 = 1e-12f;
        float mag = stddev * sqrtf(-2.0f * logf(u1));
        t->data[i] = mean + mag * cosf(2.0f * 3.14159265358979f * u2);
    }
    return t;
}

pg_tensor *pg_tensor_linspace(float start, float stop, size_t num)
{
    if (num == 0)
        return NULL;
    pg_tensor *t = pg_tensor_empty(1, &num);
    if (!t)
        return NULL;
    if (num == 1) {
        t->data[0] = start;
        return t;
    }
    float step = (stop - start) / (float)(num - 1);
    for (size_t i = 0; i < num - 1; i++)
        t->data[i] = start + step * (float)i;
    t->data[num - 1] = stop;
    return t;
}

pg_tensor *pg_tensor_eye(size_t n)
{
    if (n == 0)
        return NULL;
    size_t shape[2] = {n, n};
    pg_tensor *t = pg_tensor_empty(2, shape);
    if (!t)
        return NULL;
    memset(t->data, 0, t->numel * sizeof(float));
    for (size_t i = 0; i < n; i++)
        t->data[i * n + i] = 1.0f;
    return t;
}

pg_tensor *pg_tensor_clone(const pg_tensor *t)
{
    if (!t || !t->data_raw) return NULL;
    if (t->dtype == PG_DTYPE_F32) return pg_tensor_from_data(t->ndim, t->shape, t->data);
    return pg_tensor_from_data_dtype(t->dtype, t->ndim, t->shape, t->data_raw);
}

static size_t flat_index(const pg_tensor *t, const size_t *idx)
{
    size_t off = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        assert(idx[i] < t->shape[i]);
        off += idx[i] * t->stride[i];
    }
    return off;
}

float pg_tensor_get(const pg_tensor *t, const size_t *idx)
{
    assert(t && t->data_raw && idx);
    size_t off = flat_index(t, idx);
    if (t->dtype == PG_DTYPE_F32) return t->data[off];
    if (t->dtype == PG_DTYPE_F16) return pg_f16_to_f32_scalar(t->data_u16[off]);
    if (t->dtype == PG_DTYPE_BF16) return pg_bf16_to_f32_scalar(t->data_u16[off]);
    return 0.0f;
}

void pg_tensor_set(pg_tensor *t, const size_t *idx, float value)
{
    assert(t && t->data_raw && idx);
    size_t off = flat_index(t, idx);
    if (t->dtype == PG_DTYPE_F32) t->data[off] = value;
    else if (t->dtype == PG_DTYPE_F16) t->data_u16[off] = pg_f32_to_f16_scalar(value);
    else if (t->dtype == PG_DTYPE_BF16) t->data_u16[off] = pg_f32_to_bf16_scalar(value);
}

void pg_tensor_fill(pg_tensor *t, float value)
{
    assert(t && t->data_raw);
    size_t n = t->numel;
    if (t->dtype == PG_DTYPE_F32) {
        if (n < 65536) {
            for (size_t i = 0; i < n; i++) t->data[i] = value;
        } else {
            fill_par_t ctx={t->data, value};
            pg_parallel_for(n, 65536, fill_par_fn, &ctx);
        }
    } else if (t->dtype == PG_DTYPE_F16) {
        uint16_t hv = pg_f32_to_f16_scalar(value);
        for (size_t i=0;i<n;i++) t->data_u16[i]=hv;
    } else if (t->dtype == PG_DTYPE_BF16) {
        uint16_t bv = pg_f32_to_bf16_scalar(value);
        for (size_t i=0;i<n;i++) t->data_u16[i]=bv;
    }
}

void pg_tensor_copy_from(pg_tensor *dst, const pg_tensor *src)
{
    assert(dst && src && dst->data_raw && src->data_raw);
    assert(dst->ndim == src->ndim);
    assert(memcmp(dst->shape, src->shape, dst->ndim * sizeof(size_t)) == 0);
    assert(dst->dtype == src->dtype);
    memcpy(dst->data_raw, src->data_raw, src->numel * src->elem_size);
}

bool pg_tensor_reshape(pg_tensor *t, size_t ndim, const size_t *shape)
{
    if (!valid_shape(ndim, shape))
        return false;
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++)
        numel *= shape[i];
    if (numel != t->numel)
        return false;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, t->shape, t->stride);
    return true;
}

size_t pg_shape_numel(size_t ndim, const size_t *shape)
{
    if (!valid_shape(ndim, shape))
        return 0;
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++)
        numel *= shape[i];
    return numel;
}

bool pg_shape_equal(size_t ndim_a, const size_t *shape_a, size_t ndim_b, const size_t *shape_b)
{
    if (ndim_a != ndim_b)
        return false;
    for (size_t i = 0; i < ndim_a; i++)
        if (shape_a[i] != shape_b[i])
            return false;
    return true;
}

bool pg_tensor_allclose(const pg_tensor *a, const pg_tensor *b, float rtol, float atol)
{
    assert(a && b && a->data_raw && b->data_raw);
    if (!pg_shape_equal(a->ndim, a->shape, b->ndim, b->shape))
        return false;
    if (a->dtype != b->dtype) {
        // compare as float
        for (size_t i = 0; i < a->numel; i++) {
            float av, bv;
            if (a->dtype==PG_DTYPE_F32) av=a->data[i];
            else if (a->dtype==PG_DTYPE_F16) av=pg_f16_to_f32_scalar(a->data_u16[i]);
            else av=pg_bf16_to_f32_scalar(a->data_u16[i]);
            if (b->dtype==PG_DTYPE_F32) bv=b->data[i];
            else if (b->dtype==PG_DTYPE_F16) bv=pg_f16_to_f32_scalar(b->data_u16[i]);
            else bv=pg_bf16_to_f32_scalar(b->data_u16[i]);
            float diff = fabsf(av - bv);
            if (!(diff <= atol + rtol * fabsf(bv)))
                return false;
        }
        return true;
    }
    if (a->dtype == PG_DTYPE_F32) {
        for (size_t i = 0; i < a->numel; i++) {
            float diff = fabsf(a->data[i] - b->data[i]);
            if (!(diff <= atol + rtol * fabsf(b->data[i])))
                return false;
        }
        return true;
    }
    // f16/bf16: convert to float for comparison
    for (size_t i = 0; i < a->numel; i++) {
        float av = (a->dtype==PG_DTYPE_F16)? pg_f16_to_f32_scalar(a->data_u16[i]) : pg_bf16_to_f32_scalar(a->data_u16[i]);
        float bv = (b->dtype==PG_DTYPE_F16)? pg_f16_to_f32_scalar(b->data_u16[i]) : pg_bf16_to_f32_scalar(b->data_u16[i]);
        float diff = fabsf(av - bv);
        if (!(diff <= atol + rtol * fabsf(bv))) return false;
    }
    return true;
}

static void print_rec(const pg_tensor *t, size_t dim, size_t offset, FILE *out)
{
    fputc('[', out);
    if (dim == t->ndim - 1) {
        for (size_t i = 0; i < t->shape[dim]; i++) {
            float v;
            if (t->dtype==PG_DTYPE_F32) v=t->data[offset + i];
            else if (t->dtype==PG_DTYPE_F16) v=pg_f16_to_f32_scalar(t->data_u16[offset+i]);
            else v=pg_bf16_to_f32_scalar(t->data_u16[offset+i]);
            fprintf(out, i ? ", %g" : "%g", v);
        }
    } else {
        for (size_t i = 0; i < t->shape[dim]; i++) {
            if (i)
                fprintf(out, ",\n%*s", (int)(dim + 1), "");
            print_rec(t, dim + 1, offset + i * t->stride[dim], out);
        }
    }
    fputc(']', out);
}

pg_tensor *pg_tensor_view(const pg_tensor *src) {
    if (!src || !src->data_raw || !src->data_ref) return NULL;
    pg_tensor *v = (pg_tensor*)malloc(sizeof(*v));
    if (!v) return NULL;
    memcpy(v, src, sizeof(*v));
    v->is_view = true;
    v->view_parent = (pg_tensor*)src;
    data_ref_retain(v->data_ref);
    return v;
}
pg_tensor *pg_tensor_reshape_view(const pg_tensor *src, size_t ndim, const size_t *shape) {
    if (!src || !src->data_raw || !valid_shape(ndim, shape)) return NULL;
    size_t numel = 1; for (size_t i=0;i<ndim;i++) numel*=shape[i];
    if (numel != src->numel) return NULL;
    // only allow reshape view if src is contiguous (row-major)
    size_t acc=1; for(size_t i=src->ndim;i-- >0;){ if(src->stride[i]!=acc) return NULL; acc*=src->shape[i]; }
    pg_tensor *v = pg_tensor_view(src);
    if (!v) return NULL;
    v->ndim = ndim;
    memcpy(v->shape, shape, ndim*sizeof(size_t));
    compute_strides(ndim, shape, v->stride);
    return v;
}
pg_tensor *pg_tensor_permute_view(const pg_tensor *src, const size_t *order) {
    if (!src || !src->data_raw || !order) return NULL;
    for(size_t i=0;i<src->ndim;i++) if(order[i]>=src->ndim) return NULL;
    // check permutation is bijection
    bool seen[PG_MAX_NDIM]={0};
    for(size_t i=0;i<src->ndim;i++){ if(seen[order[i]]) return NULL; seen[order[i]]=true; }
    pg_tensor *v = pg_tensor_view(src);
    if (!v) return NULL;
    size_t nshape[PG_MAX_NDIM], nstride[PG_MAX_NDIM];
    for(size_t i=0;i<src->ndim;i++){ nshape[i]=src->shape[order[i]]; nstride[i]=src->stride[order[i]]; }
    memcpy(v->shape, nshape, src->ndim*sizeof(size_t));
    memcpy(v->stride, nstride, src->ndim*sizeof(size_t));
    return v;
}

void pg_tensor_print(const pg_tensor *t, FILE *out)
{
    assert(t && t->data_raw && out);
    print_rec(t, 0, 0, out);
    fputc('\n', out);
}

// ---- serialization ----
#define PG_TENSOR_MAGIC "PGT1"
#define PG_TENSOR_VERSION 1

bool pg_tensor_save_fp(const pg_tensor *t, FILE *fp){
    if(!t || !t->data_raw || !fp) return false;
    // header: magic 4 + version 4 + dtype 4 + ndim 4 + shape[ndim] *8 + numel 8 + elem_size 8 + nbytes 8
    if(fwrite(PG_TENSOR_MAGIC, 1, 4, fp) != 4) return false;
    uint32_t ver = PG_TENSOR_VERSION;
    if(fwrite(&ver, 4, 1, fp) != 1) return false;
    uint32_t dtype = (uint32_t)t->dtype;
    if(fwrite(&dtype, 4, 1, fp) != 1) return false;
    uint32_t ndim = (uint32_t)t->ndim;
    if(fwrite(&ndim, 4, 1, fp) != 1) return false;
    for(size_t i=0;i<t->ndim;i++){
        uint64_t d = (uint64_t)t->shape[i];
        if(fwrite(&d, 8, 1, fp) != 1) return false;
    }
    uint64_t numel = (uint64_t)t->numel;
    if(fwrite(&numel, 8, 1, fp) != 1) return false;
    uint64_t elem = (uint64_t)t->elem_size;
    if(fwrite(&elem, 8, 1, fp) != 1) return false;
    size_t nbytes = t->numel * t->elem_size;
    // ensure contiguous for save: copy logically contiguous data in row-major order
    // If tensor is contiguous we can dump raw, else iterate.
    bool contig = true;
    size_t acc=1;
    for(size_t i=t->ndim;i-- >0;){
        if(t->stride[i]!=acc) { contig=false; break; }
        acc*=t->shape[i];
    }
    if(contig){
        if(nbytes && fwrite(t->data_raw, 1, nbytes, fp) != nbytes) return false;
    } else {
        // fallback: materialize contiguous buffer and write
        // iterate via odometer and write elements in logical order as packed
        size_t idx[PG_MAX_NDIM]={0};
        for(size_t p=0;p<t->numel;p++){
            size_t off=0;
            for(size_t d=0;d<t->ndim;d++) off += idx[d]*t->stride[d];
            if(fwrite((char*)t->data_raw + off*t->elem_size, t->elem_size, 1, fp) != 1) return false;
            for(size_t d=t->ndim; d-- >0;){
                idx[d]++;
                if(idx[d] < t->shape[d]) break;
                idx[d]=0;
            }
        }
    }
    return true;
}

bool pg_tensor_save(const pg_tensor *t, const char *path){
    if(!path) return false;
    FILE *fp = fopen(path, "wb");
    if(!fp) return false;
    bool ok = pg_tensor_save_fp(t, fp);
    fclose(fp);
    return ok;
}

pg_tensor *pg_tensor_load_fp(FILE *fp){
    if(!fp) return NULL;
    char magic[4];
    if(fread(magic, 1, 4, fp) != 4) return NULL;
    if(memcmp(magic, PG_TENSOR_MAGIC, 4) != 0) return NULL;
    uint32_t ver=0, dtype_u=0, ndim_u=0;
    if(fread(&ver, 4, 1, fp) != 1) return NULL;
    if(ver != PG_TENSOR_VERSION) return NULL;
    if(fread(&dtype_u, 4, 1, fp) != 1) return NULL;
    if(fread(&ndim_u, 4, 1, fp) != 1) return NULL;
    if(ndim_u==0 || ndim_u>PG_MAX_NDIM) return NULL;
    pg_dtype dtype = (pg_dtype)dtype_u;
    if(dtype >= PG_DTYPE_COUNT) return NULL;
    size_t ndim = (size_t)ndim_u;
    size_t shape[PG_MAX_NDIM]={0};
    for(size_t i=0;i<ndim;i++){
        uint64_t d=0;
        if(fread(&d, 8, 1, fp) != 1) return NULL;
        if(d==0 || d > 10000000) return NULL; // sanity
        shape[i]=(size_t)d;
    }
    uint64_t numel_u=0, elem_u=0;
    if(fread(&numel_u, 8, 1, fp) != 1) return NULL;
    if(fread(&elem_u, 8, 1, fp) != 1) return NULL;
    size_t expected = pg_shape_numel(ndim, shape);
    if(expected != (size_t)numel_u) return NULL;
    if(pg_dtype_size(dtype) != (size_t)elem_u) return NULL;
    pg_tensor *t = pg_tensor_empty_dtype(dtype, ndim, shape);
    if(!t) return NULL;
    size_t nbytes = t->numel * t->elem_size;
    if(nbytes){
        if(fread(t->data_raw, 1, nbytes, fp) != nbytes){
            pg_tensor_free(t);
            return NULL;
        }
    }
    return t;
}

pg_tensor *pg_tensor_load(const char *path){
    if(!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if(!fp) return NULL;
    pg_tensor *t = pg_tensor_load_fp(fp);
    fclose(fp);
    return t;
}
