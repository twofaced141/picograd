#include "clip.h"
#include <math.h>
#include <assert.h>

float pg_grad_norm(pg_node **params, size_t n){
    if(!params || n==0) return 0.0f;
    double sum = 0.0;
    for(size_t i=0;i<n;i++){
        pg_node *p = params[i];
        if(!p || !p->grad || !p->grad->data) continue;
        pg_tensor *g = p->grad;
        for(size_t j=0;j<g->numel;j++){
            double v = (double)g->data[j];
            sum += v*v;
        }
    }
    return (float)sqrt(sum);
}

float pg_grad_norm_module(pg_module *m){
    if(!m) return 0.0f;
    size_t n = pg_module_num_params(m);
    pg_node **p = pg_module_params(m);
    return pg_grad_norm(p, n);
}

float pg_clip_grad_norm(pg_node **params, size_t n, float max_norm, float eps){
    if(!params || max_norm <= 0.0f) return -1.0f;
    if(eps < 0) eps = 1e-6f;
    float total = pg_grad_norm(params, n);
    if(total <= max_norm) return total;
    float scale = max_norm / (total + eps);
    for(size_t i=0;i<n;i++){
        pg_node *p = params[i];
        if(!p || !p->grad || !p->grad->data) continue;
        pg_tensor *g = p->grad;
        for(size_t j=0;j<g->numel;j++) g->data[j] *= scale;
    }
    return total;
}

float pg_clip_grad_norm_module(pg_module *m, float max_norm, float eps){
    if(!m) return -1.0f;
    size_t n = pg_module_num_params(m);
    pg_node **p = pg_module_params(m);
    return pg_clip_grad_norm(p, n, max_norm, eps);
}

void pg_clip_grad_value(pg_node **params, size_t n, float clip_value){
    if(!params || clip_value <= 0.0f) return;
    for(size_t i=0;i<n;i++){
        pg_node *p = params[i];
        if(!p || !p->grad || !p->grad->data) continue;
        pg_tensor *g = p->grad;
        for(size_t j=0;j<g->numel;j++){
            float v = g->data[j];
            if(v > clip_value) g->data[j] = clip_value;
            else if(v < -clip_value) g->data[j] = -clip_value;
        }
    }
}

void pg_clip_grad_value_module(pg_module *m, float clip_value){
    if(!m) return;
    size_t n = pg_module_num_params(m);
    pg_node **p = pg_module_params(m);
    pg_clip_grad_value(p, n, clip_value);
}
