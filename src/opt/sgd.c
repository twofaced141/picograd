#include "sgd.h"

#include <assert.h>
#include <stdlib.h>

typedef struct {
    pg_node *param;
    pg_tensor *buf;
} pg_sgd_state;

struct pg_sgd {
    pg_sgd_cfg cfg;
    pg_sgd_state *states;
    size_t nparams;
    size_t cap;
};

pg_sgd_cfg pg_sgd_cfg_default(void)
{
    pg_sgd_cfg c = {
        .lr = 0.01f,
        .momentum = 0.0f,
        .dampening = 0.0f,
        .weight_decay = 0.0f,
        .nesterov = false,
    };
    return c;
}

pg_sgd *pg_sgd_new(const pg_sgd_cfg *cfg)
{
    static const pg_sgd_cfg def = {
        .lr = 0.01f,
        .momentum = 0.0f,
        .dampening = 0.0f,
        .weight_decay = 0.0f,
        .nesterov = false,
    };

    if (!cfg)
        cfg = &def;

    assert(cfg->lr >= 0.0f);
    assert(cfg->momentum >= 0.0f);
    assert(cfg->dampening >= 0.0f);
    assert(cfg->weight_decay >= 0.0f);
    assert(!cfg->nesterov || (cfg->momentum > 0.0f && cfg->dampening == 0.0f));

    pg_sgd *o = malloc(sizeof(*o));
    if (!o)
        return NULL;

    o->cfg = *cfg;
    o->states = NULL;
    o->nparams = 0;
    o->cap = 0;
    return o;
}

void pg_sgd_free(pg_sgd *opt)
{
    if (!opt)
        return;
    for (size_t i = 0; i < opt->nparams; i++) {
        pg_node_free(opt->states[i].param);
        pg_tensor_free(opt->states[i].buf);
    }
    free(opt->states);
    free(opt);
}

int pg_sgd_add_param(pg_sgd *opt, pg_node *param)
{
    assert(opt && param && param->requires_grad);

    if (opt->nparams == opt->cap) {
        size_t ncap = opt->cap ? opt->cap * 2 : 8;
        pg_sgd_state *ns = realloc(opt->states, ncap * sizeof(*opt->states));
        if (!ns)
            return -1;
        opt->states = ns;
        opt->cap = ncap;
    }

    if (!pg_node_retain(param))
        return -1;

    opt->states[opt->nparams].param = param;
    opt->states[opt->nparams].buf = NULL;
    opt->nparams++;
    return 0;
}

size_t pg_sgd_num_params(const pg_sgd *opt)
{
    return opt ? opt->nparams : 0;
}

void pg_sgd_zero_grad(pg_sgd *opt)
{
    assert(opt);
    for (size_t i = 0; i < opt->nparams; i++) {
        pg_node *p = opt->states[i].param;
        if (p->grad) {
            pg_tensor_free(p->grad);
            p->grad = NULL;
        }
    }
}

void pg_sgd_step(pg_sgd *opt)
{
    assert(opt);

    const float lr = opt->cfg.lr;
    const float mu = opt->cfg.momentum;
    const float damp = opt->cfg.dampening;
    const float wd = opt->cfg.weight_decay;
    const bool nesterov = opt->cfg.nesterov;

    for (size_t i = 0; i < opt->nparams; i++) {
        pg_node *p = opt->states[i].param;
        pg_tensor *g = p->grad;
        if (!g)
            continue;

        assert(g->numel == p->value->numel);

        float *pv = p->value->data;
        float *gv = g->data;

        pg_tensor *buf = NULL;
        float *bv = NULL;
        if (mu != 0.0f) {
            buf = opt->states[i].buf;
            if (!buf) {
                buf = pg_tensor_zeros(p->value->ndim, p->value->shape);
                if (!buf) {
                    assert(buf);
                    continue;
                }
                opt->states[i].buf = buf;
            }
            bv = buf->data;
        }

        for (size_t j = 0; j < p->value->numel; j++) {
            float gj = gv[j];

            if (wd != 0.0f)
                gj += wd * pv[j];

            float update = gj;
            if (mu != 0.0f) {
                bv[j] = bv[j] * mu + gj * (1.0f - damp);
                update = nesterov ? gj + mu * bv[j] : bv[j];
            }

            pv[j] -= lr * update;
        }
    }
}
