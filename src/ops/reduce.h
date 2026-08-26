#ifndef PICOGRAD_OPS_REDUCE_H
#define PICOGRAD_OPS_REDUCE_H

#include "../core/tensor.h"

#include <stdbool.h>

pg_tensor *pg_sum(const pg_tensor *t, size_t axis, bool keepdim);
pg_tensor *pg_mean(const pg_tensor *t, size_t axis, bool keepdim);
pg_tensor *pg_max(const pg_tensor *t, size_t axis, bool keepdim);
pg_tensor *pg_min(const pg_tensor *t, size_t axis, bool keepdim);

pg_tensor *pg_var(const pg_tensor *t, size_t axis, bool keepdim, int ddof);
pg_tensor *pg_std(const pg_tensor *t, size_t axis, bool keepdim, int ddof);

pg_tensor *pg_argmax(const pg_tensor *t, size_t axis, bool keepdim);
pg_tensor *pg_argmin(const pg_tensor *t, size_t axis, bool keepdim);

#endif
