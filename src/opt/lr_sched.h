#ifndef PICOGRAD_OPT_LR_SCHED_H
#define PICOGRAD_OPT_LR_SCHED_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PG_SCHED_CONSTANT,
    PG_SCHED_COSINE,   // linear warmup then cosine decay to min_lr
    PG_SCHED_LINEAR,   // linear decay base_lr -> min_lr over total steps
    PG_SCHED_STEP      // base_lr * gamma^(floor(step/step_size))
} pg_lr_sched_kind;

typedef struct pg_lr_sched pg_lr_sched;

// cosine: warmup from ~0 to base_lr over `warmup` steps, then decay to
// min_lr over the remaining (total - warmup) steps.
pg_lr_sched *pg_lr_sched_new_cosine(float base_lr, float min_lr,
                                    size_t warmup, size_t total);
pg_lr_sched *pg_lr_sched_new_linear(float base_lr, float min_lr, size_t total);
pg_lr_sched *pg_lr_sched_new_step(float base_lr, float gamma, size_t step_size);
pg_lr_sched *pg_lr_sched_new_constant(float base_lr);

// current learning rate (does not advance the step counter)
float pg_lr_sched_current(const pg_lr_sched *s);

// advance one step and return the new learning rate
float pg_lr_sched_step(pg_lr_sched *s);

void pg_lr_sched_reset(pg_lr_sched *s);
void pg_lr_sched_free(pg_lr_sched *s);

#ifdef __cplusplus
}
#endif

#endif
