#ifndef PICOGRAD_OPS_INDEX_H
#define PICOGRAD_OPS_INDEX_H

#include "../core/tensor.h"

pg_tensor *pg_gather(const pg_tensor *t, size_t axis, const pg_tensor *indices);
pg_tensor *pg_scatter(const pg_tensor *t, size_t axis, const pg_tensor *indices, const pg_tensor *src);
pg_tensor *pg_index_select(const pg_tensor *t, size_t axis, const pg_tensor *indices);
pg_tensor *pg_masked_select(const pg_tensor *t, const pg_tensor *mask);

#endif
