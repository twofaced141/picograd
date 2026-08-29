#ifndef PICOGRAD_OPT_ADAM_H
#define PICOGRAD_OPT_ADAM_H

#include "../autograd/autograd.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    bool amsgrad;
    bool decoupled_wd; // true = AdamW, false = Adam L2
} pg_adam_cfg;

pg_adam_cfg pg_adam_cfg_default(void);

typedef struct pg_adam pg_adam;

pg_adam *pg_adam_new(const pg_adam_cfg *cfg);
void pg_adam_free(pg_adam *opt);

int pg_adam_add_param(pg_adam *opt, pg_node *param);
// per-parameter lr / weight_decay override. lr<=0 or wd<0 -> use cfg defaults.
int pg_adam_add_param_lr(pg_adam *opt, pg_node *param, float lr, float wd);
size_t pg_adam_num_params(const pg_adam *opt);

void pg_adam_set_lr(pg_adam *opt, float lr);
float pg_adam_get_lr(const pg_adam *opt);
void pg_adam_set_param_lr(pg_adam *opt, size_t idx, float lr);
void pg_adam_set_param_wd(pg_adam *opt, size_t idx, float wd);

void pg_adam_zero_grad(pg_adam *opt);
void pg_adam_step(pg_adam *opt);

// optional: set step counter manually (for checkpointing)
void pg_adam_set_step(pg_adam *opt, unsigned long long step);
unsigned long long pg_adam_get_step(const pg_adam *opt);

#ifdef __cplusplus
}
#endif

#endif
