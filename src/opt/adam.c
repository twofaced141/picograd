#include "adam.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    pg_node *param;
    pg_tensor *m;
    pg_tensor *v;
    pg_tensor *vmax; // for amsgrad
    float lr;        // per-param override (0 = use cfg)
    float wd;        // per-param override (<0 = use cfg)
} adam_state_t;

struct pg_adam {
    pg_adam_cfg cfg;
    adam_state_t *states;
    size_t nparams;
    size_t cap;
    unsigned long long step;
};

pg_adam_cfg pg_adam_cfg_default(void){
    pg_adam_cfg c;
    c.lr = 1e-3f;
    c.beta1 = 0.9f;
    c.beta2 = 0.999f;
    c.eps = 1e-8f;
    c.weight_decay = 0.0f;
    c.amsgrad = false;
    c.decoupled_wd = true; // AdamW by default
    return c;
}

pg_adam *pg_adam_new(const pg_adam_cfg *cfg){
    static const pg_adam_cfg def = {1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, true};
    if(!cfg) cfg=&def;
    assert(cfg->lr >=0);
    assert(cfg->beta1 >=0 && cfg->beta1 <1);
    assert(cfg->beta2 >=0 && cfg->beta2 <1);
    assert(cfg->eps >0);
    assert(cfg->weight_decay >=0);
    pg_adam *o=malloc(sizeof(*o));
    if(!o) return NULL;
    o->cfg=*cfg;
    o->states=NULL;
    o->nparams=0;
    o->cap=0;
    o->step=0;
    return o;
}

void pg_adam_free(pg_adam *opt){
    if(!opt) return;
    for(size_t i=0;i<opt->nparams;i++){
        pg_node_free(opt->states[i].param);
        pg_tensor_free(opt->states[i].m);
        pg_tensor_free(opt->states[i].v);
        pg_tensor_free(opt->states[i].vmax);
    }
    free(opt->states);
    free(opt);
}

int pg_adam_add_param(pg_adam *opt, pg_node *param){
    return pg_adam_add_param_lr(opt, param, 0.0f, -1.0f);
}

int pg_adam_add_param_lr(pg_adam *opt, pg_node *param, float lr, float wd){
    assert(opt && param && param->requires_grad);
    if(opt->nparams==opt->cap){
        size_t ncap=opt->cap?opt->cap*2:8;
        adam_state_t *ns=realloc(opt->states, ncap*sizeof(*ns));
        if(!ns) return -1;
        opt->states=ns;
        opt->cap=ncap;
    }
    if(!pg_node_retain(param)) return -1;
    opt->states[opt->nparams].param=param;
    opt->states[opt->nparams].m=NULL;
    opt->states[opt->nparams].v=NULL;
    opt->states[opt->nparams].vmax=NULL;
    opt->states[opt->nparams].lr=lr;
    opt->states[opt->nparams].wd=wd;
    opt->nparams++;
    return 0;
}

size_t pg_adam_num_params(const pg_adam *opt){ return opt?opt->nparams:0; }

void pg_adam_set_lr(pg_adam *opt, float lr){ if(opt) opt->cfg.lr=lr; }
float pg_adam_get_lr(const pg_adam *opt){ return opt?opt->cfg.lr:0.0f; }
void pg_adam_set_param_lr(pg_adam *opt, size_t idx, float lr){
    assert(opt && idx<opt->nparams); opt->states[idx].lr=lr;
}
void pg_adam_set_param_wd(pg_adam *opt, size_t idx, float wd){
    assert(opt && idx<opt->nparams); opt->states[idx].wd=wd;
}

void pg_adam_zero_grad(pg_adam *opt){
    assert(opt);
    for(size_t i=0;i<opt->nparams;i++){
        pg_node *p=opt->states[i].param;
        if(p->grad){ pg_tensor_free(p->grad); p->grad=NULL; }
    }
}

void pg_adam_set_step(pg_adam *opt, unsigned long long s){ if(opt) opt->step=s; }
unsigned long long pg_adam_get_step(const pg_adam *opt){ return opt?opt->step:0; }

void pg_adam_step(pg_adam *opt){
    assert(opt);
    float b1=opt->cfg.beta1;
    float b2=opt->cfg.beta2;
    float eps=opt->cfg.eps;
    bool amsgrad=opt->cfg.amsgrad;
    bool decoupled=opt->cfg.decoupled_wd;
    opt->step++;
    float bc1 = 1.0f - powf(b1, (float)opt->step);
    float bc2 = 1.0f - powf(b2, (float)opt->step);
    // clamp bc to avoid div by zero early
    if(bc1 < 1e-12f) bc1=1e-12f;
    if(bc2 < 1e-12f) bc2=1e-12f;

    for(size_t i=0;i<opt->nparams;i++){
        pg_node *p=opt->states[i].param;
        pg_tensor *g=p->grad;
        if(!g) continue;
        assert(g->numel==p->value->numel);
        float *pv=p->value->data;
        float *gv=g->data;
        size_t n=p->value->numel;

        // per-param lr / wd override (0 / <0 => use cfg)
        float lr = opt->states[i].lr > 0.0f ? opt->states[i].lr : opt->cfg.lr;
        float wd = opt->states[i].wd >= 0.0f ? opt->states[i].wd : opt->cfg.weight_decay;

        // lazy init m/v
        if(!opt->states[i].m){
            opt->states[i].m=pg_tensor_zeros(p->value->ndim, p->value->shape);
            opt->states[i].v=pg_tensor_zeros(p->value->ndim, p->value->shape);
            if(amsgrad) opt->states[i].vmax=pg_tensor_zeros(p->value->ndim, p->value->shape);
            if(!opt->states[i].m || !opt->states[i].v || (amsgrad && !opt->states[i].vmax)) {
                pg_tensor_free(opt->states[i].m); opt->states[i].m=NULL;
                pg_tensor_free(opt->states[i].v); opt->states[i].v=NULL;
                pg_tensor_free(opt->states[i].vmax); opt->states[i].vmax=NULL;
                assert(0);
                continue;
            }
        }
        float *m=opt->states[i].m->data;
        float *v=opt->states[i].v->data;
        float *vmax = amsgrad ? opt->states[i].vmax->data : NULL;

        // decoupled weight decay (AdamW): p -= lr*wd*p  before moment update
        if(wd!=0.0f && decoupled){
            for(size_t j=0;j<n;j++) pv[j] -= lr * wd * pv[j];
        }

        for(size_t j=0;j<n;j++){
            float gj=gv[j];
            if(wd!=0.0f && !decoupled){
                gj += wd * pv[j];
            }
            m[j] = b1 * m[j] + (1.0f - b1) * gj;
            v[j] = b2 * v[j] + (1.0f - b2) * gj * gj;
            float m_hat = m[j] / bc1;
            float v_hat;
            if(amsgrad){
                if(v[j] > vmax[j]) vmax[j]=v[j];
                v_hat = vmax[j] / bc2;
            } else {
                v_hat = v[j] / bc2;
            }
            float denom = sqrtf(v_hat) + eps;
            pv[j] -= lr * m_hat / denom;
        }
    }
}
