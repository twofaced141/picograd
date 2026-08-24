#ifndef PICOGRAD_CORE_TENSOR_H
#define PICOGRAD_CORE_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PG_MAX_NDIM 8

typedef struct pg_tensor {
    size_t ndim;
    size_t shape[PG_MAX_NDIM];
    size_t stride[PG_MAX_NDIM];
    size_t numel;
    float *data;
} pg_tensor;

pg_tensor *pg_tensor_new(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_zeros(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_ones(size_t ndim, const size_t *shape);
pg_tensor *pg_tensor_full(size_t ndim, const size_t *shape, float value);
pg_tensor *pg_tensor_from_data(size_t ndim, const size_t *shape, const float *data);
pg_tensor *pg_tensor_arange(float start, float stop, float step);
pg_tensor *pg_tensor_uniform(size_t ndim, const size_t *shape, float low, float high);
pg_tensor *pg_tensor_normal(size_t ndim, const size_t *shape, float mean, float stddev);

pg_tensor *pg_tensor_clone(const pg_tensor *t);
void pg_tensor_free(pg_tensor *t);

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

#ifdef __cplusplus
}
#endif

#endif
