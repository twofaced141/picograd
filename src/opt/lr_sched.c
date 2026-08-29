#include "lr_sched.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

struct pg_lr_sched {
    pg_lr_sched_kind kind;
    float base_lr;
    float min_lr;
    size_t warmup;
    size_t total;
    size_t step_size;
    float gamma;
    unsigned long long step;
};

pg_lr_sched *pg_lr_sched_new_cosine(float base_lr, float min_lr,
                                    size_t warmup, size_t total)
{
    assert(base_lr >= 0 && min_lr >= 0);
    pg_lr_sched *s = malloc(sizeof(*s));
    if (!s)
        return NULL;
    s->kind = PG_SCHED_COSINE;
    s->base_lr = base_lr;
    s->min_lr = min_lr;
    s->warmup = warmup;
    s->total = total ? total : 1;
    s->step = 0;
    return s;
}

pg_lr_sched *pg_lr_sched_new_linear(float base_lr, float min_lr, size_t total)
{
    assert(base_lr >= 0 && min_lr >= 0 && total > 0);
    pg_lr_sched *s = malloc(sizeof(*s));
    if (!s)
        return NULL;
    s->kind = PG_SCHED_LINEAR;
    s->base_lr = base_lr;
    s->min_lr = min_lr;
    s->total = total;
    s->step = 0;
    return s;
}

pg_lr_sched *pg_lr_sched_new_step(float base_lr, float gamma, size_t step_size)
{
    assert(base_lr >= 0 && gamma >= 0 && step_size > 0);
    pg_lr_sched *s = malloc(sizeof(*s));
    if (!s)
        return NULL;
    s->kind = PG_SCHED_STEP;
    s->base_lr = base_lr;
    s->min_lr = 0.0f;
    s->gamma = gamma;
    s->step_size = step_size;
    s->step = 0;
    return s;
}

pg_lr_sched *pg_lr_sched_new_constant(float base_lr)
{
    assert(base_lr >= 0);
    pg_lr_sched *s = malloc(sizeof(*s));
    if (!s)
        return NULL;
    s->kind = PG_SCHED_CONSTANT;
    s->base_lr = base_lr;
    s->min_lr = base_lr;
    s->step = 0;
    return s;
}

float pg_lr_sched_current(const pg_lr_sched *s)
{
    assert(s);
    switch (s->kind) {
    case PG_SCHED_CONSTANT:
        return s->base_lr;
    case PG_SCHED_STEP:
        return s->base_lr * powf(s->gamma, (float)(s->step / s->step_size));
    case PG_SCHED_LINEAR: {
        float frac = (float)(s->step + 1) / (float)s->total;
        if (frac > 1.0f)
            frac = 1.0f;
        return s->base_lr + (s->min_lr - s->base_lr) * frac;
    }
    case PG_SCHED_COSINE: {
        if (s->step < s->warmup) {
            if (s->warmup == 0)
                return s->base_lr;
            float f = (float)(s->step + 1) / (float)s->warmup;
            return s->base_lr * f;
        }
        size_t decay = s->step - s->warmup;
        size_t decay_total = s->total > s->warmup ? s->total - s->warmup : 1;
        float frac = (float)decay / (float)decay_total;
        if (frac > 1.0f)
            frac = 1.0f;
        float c = 0.5f * (1.0f + cosf(3.14159265358979323846f * frac));
        return s->min_lr + (s->base_lr - s->min_lr) * c;
    }
    }
    return s->base_lr;
}

float pg_lr_sched_step(pg_lr_sched *s)
{
    assert(s);
    float lr = pg_lr_sched_current(s);
    s->step++;
    return lr;
}

void pg_lr_sched_reset(pg_lr_sched *s)
{
    if (s)
        s->step = 0;
}

void pg_lr_sched_free(pg_lr_sched *s)
{
    free(s);
}
