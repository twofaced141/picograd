#ifndef PICOGRAD_OPS_MATMUL_H
#define PICOGRAD_OPS_MATMUL_H

#include "../core/tensor.h"

pg_tensor *pg_matmul(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_bmm(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_addmm(const pg_tensor *input, const pg_tensor *m1, const pg_tensor *m2,
                    float alpha, float beta);
pg_tensor *pg_tensordot(const pg_tensor *a, const pg_tensor *b,
                        size_t ndims, const size_t *axes_a, const size_t *axes_b);

#endif
