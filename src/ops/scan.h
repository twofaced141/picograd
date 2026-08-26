#ifndef PICOGRAD_OPS_SCAN_H
#define PICOGRAD_OPS_SCAN_H

#include "../core/tensor.h"

#include <stdbool.h>

typedef struct {
    pg_tensor *values;
    pg_tensor *indices;
} pg_kv;

pg_tensor *pg_cumsum(const pg_tensor *t, size_t axis);
pg_tensor *pg_cumprod(const pg_tensor *t, size_t axis);

pg_kv pg_sort(const pg_tensor *t, size_t axis, bool descending);
pg_kv pg_topk(const pg_tensor *t, size_t axis, size_t k);

#endif
