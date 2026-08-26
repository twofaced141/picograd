#ifndef PICOGRAD_OPS_ELEMENTWISE_H
#define PICOGRAD_OPS_ELEMENTWISE_H

#include "../core/tensor.h"

pg_tensor *pg_add(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_sub(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_mul(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_div(const pg_tensor *a, const pg_tensor *b);
pg_tensor *pg_pow(const pg_tensor *a, const pg_tensor *b);

pg_tensor *pg_neg(const pg_tensor *a);
pg_tensor *pg_abs(const pg_tensor *a);
pg_tensor *pg_sqrt(const pg_tensor *a);
pg_tensor *pg_clamp(const pg_tensor *a, float lo, float hi);

pg_tensor *pg_exp(const pg_tensor *a);
pg_tensor *pg_log(const pg_tensor *a);
pg_tensor *pg_sin(const pg_tensor *a);
pg_tensor *pg_cos(const pg_tensor *a);
pg_tensor *pg_erf(const pg_tensor *a);

#endif
