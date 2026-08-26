#ifndef PICOGRAD_OPT_SGD_H
#define PICOGRAD_OPT_SGD_H

#include "../autograd/autograd.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;
} pg_sgd_cfg;

pg_sgd_cfg pg_sgd_cfg_default(void);

typedef struct pg_sgd pg_sgd;

pg_sgd *pg_sgd_new(const pg_sgd_cfg *cfg);
void pg_sgd_free(pg_sgd *opt);

int pg_sgd_add_param(pg_sgd *opt, pg_node *param);
size_t pg_sgd_num_params(const pg_sgd *opt);

void pg_sgd_zero_grad(pg_sgd *opt);
void pg_sgd_step(pg_sgd *opt);

#ifdef __cplusplus
}
#endif

#endif
