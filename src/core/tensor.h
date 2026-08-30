#ifndef PICOGRAD_CORE_TENSOR_H
#define PICOGRAD_CORE_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PG_MAX_NDIM 8

typedef struct pg_data_ref {
    void *ptr;
    size_t nbytes;
    int refs;
} pg_data_ref;

typedef struct pg_tensor {
    size_t ndim;
    size_t shape[PG_MAX_NDIM];
    size_t stride[PG_MAX_NDIM];
    size_t numel;
    union {
        float *data;       // legacy f32 accessor (valid when dtype==PG_DTYPE_F32)
        void *data_raw;    // generic accessor
        uint16_t *data_u16;// for f16/bf16
    };
    pg_data_ref *data_ref;
    bool is_view;
    struct pg_tensor *view_parent;
    pg_dtype dtype;
    size_t elem_size;
} pg_tensor;

// generic dtype-aware constructors
pg_tensor *pg_tensor_new_dtype(pg_dtype dtype, size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_empty_dtype(pg_dtype dtype, size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_from_data_dtype(pg_dtype dtype, size_t ndim, const size_t *shape, const void *data);
pg_tensor *pg_tensor_zeros_dtype(pg_dtype dtype, size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_ones_dtype(pg_dtype dtype, size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_full_dtype(pg_dtype dtype, size_t ndim, const size_t *shape, float value);

pg_tensor *pg_tensor_new(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_empty(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_zeros(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_ones(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_full(size_t ndim, const size_t *shape, float value);
pg_tensor *pg_tensor_from_data(size_t ndim, const size_t *shape, const float *data);
pg_tensor *pg_tensor_arange(float start, float stop, float step);
pg_tensor *pg_tensor_uniform(size_t ndim, const size_t *shape, float low, float high);
pg_tensor *pg_tensor_normal(size_t ndim, const size_t *shape, float mean, float stddev);
pg_tensor *pg_tensor_linspace(float start, float stop, size_t num);
pg_tensor *pg_tensor_eye(size_t n);

pg_tensor *pg_tensor_clone(const pg_tensor *t);
void pg_tensor_free(pg_tensor *t);

static inline float *pg_tensor_f32(const pg_tensor *t){ return (float*)t->data_raw; }
static inline uint16_t *pg_tensor_f16(const pg_tensor *t){ return (uint16_t*)t->data_raw; }
static inline uint16_t *pg_tensor_bf16(const pg_tensor *t){ return (uint16_t*)t->data_raw; }
static inline size_t pg_tensor_nbytes(const pg_tensor *t){ return t->numel * t->elem_size; }

void pg_seed(unsigned long long seed);

float pg_tensor_get(const pg_tensor *t, const size_t *idx);
void pg_tensor_set(pg_tensor *t, const size_t *idx, float value);

void pg_tensor_fill(pg_tensor *t, float value);
void pg_tensor_copy_from(pg_tensor *dst, const pg_tensor *src);
bool pg_tensor_reshape(pg_tensor *t, size_t ndim, const size_t *shape);

size_t pg_shape_numel(size_t ndim, const size_t *shape);
bool pg_shape_equal(size_t ndim_a, const size_t *shape_a, size_t ndim_b, const size_t *shape_b);

bool pg_tensor_allclose(const pg_tensor *a, const pg_tensor *b, float rtol, float atol);
void pg_tensor_print(const pg_tensor *t, FILE *out);

// view (no copy, shared data) — reference counted, parent may be freed before view
pg_tensor *pg_tensor_view(const pg_tensor *src);
pg_tensor *pg_tensor_reshape_view(const pg_tensor *src, size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_permute_view(const pg_tensor *src, const size_t *order);

// buffer pool (aligned 64B)
void pg_tensor_pool_clear(void);
size_t pg_tensor_pool_size(void);
size_t pg_tensor_pool_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
