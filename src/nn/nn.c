#include "nn.h"

#include "../backend/backend.h"
#include "../backend/cpu/gemm.h"
#include "../ops/matmul.h"
#include "../ops/elementwise.h"
#include "../ops/activations.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

pg_linear *pg_linear_new(size_t in_features, size_t out_features)
{
    assert(in_features > 0 && out_features > 0);

    pg_linear *l = malloc(sizeof(*l));
    if (!l)
        return NULL;

    size_t wshape[2] = {in_features, out_features};
    float limit = sqrtf(6.0f / (float)(in_features + out_features));

    l->weight = pg_tensor_uniform(2, wshape, -limit, limit);
    l->bias = pg_tensor_zeros(1, &out_features);
    if (!l->weight || !l->bias) {
        pg_tensor_free(l->weight);
        pg_tensor_free(l->bias);
        free(l);
        return NULL;
    }

    l->in_features = in_features;
    l->out_features = out_features;
    return l;
}

void pg_linear_free(pg_linear *layer)
{
    if (!layer)
        return;
    pg_tensor_free(layer->weight);
    pg_tensor_free(layer->bias);
    free(layer);
}

pg_tensor *pg_linear_forward(const pg_linear *layer, const pg_tensor *x)
{
    assert(layer && x && x->data);
    assert(x->ndim == 2);
    assert(x->shape[1] == layer->in_features);

    // fused path for CPU 2D (avoids extra alloc + add)
    if (pg_get_device() == PG_DEV_CPU) {
        size_t m = x->shape[0];
        size_t k = layer->in_features;
        size_t n = layer->out_features;
        size_t out_shape[2] = {m, n};
        pg_tensor *out = pg_tensor_empty(2, out_shape);
        if (!out) return NULL;
        pg_cpu_gemm_fused(m, n, k, x->data, k, layer->weight->data, n, out->data, n, layer->bias->data, PG_ACT_NONE);
        return out;
    }

    pg_tensor *y = pg_matmul(x, layer->weight);
    if (!y)
        return NULL;

    pg_tensor *out = pg_add(y, layer->bias);
    pg_tensor_free(y);
    return out;
}

pg_tensor *pg_linear_forward_relu(const pg_linear *layer, const pg_tensor *x)
{
    assert(layer && x && x->data);
    assert(x->ndim == 2);
    assert(x->shape[1] == layer->in_features);
    if (pg_get_device() == PG_DEV_CPU) {
        size_t m = x->shape[0];
        size_t k = layer->in_features;
        size_t n = layer->out_features;
        size_t out_shape[2] = {m, n};
        pg_tensor *out = pg_tensor_empty(2, out_shape);
        if (!out) return NULL;
        pg_cpu_gemm_fused(m, n, k, x->data, k, layer->weight->data, n, out->data, n, layer->bias->data, PG_ACT_RELU);
        return out;
    }
    pg_tensor *y = pg_linear_forward(layer, x);
    if (!y) return NULL;
    pg_tensor *out = pg_relu(y);
    pg_tensor_free(y);
    return out;
}

/* ---- Autograd Linear ---- */
pg_ag_linear *pg_ag_linear_new(size_t in_features, size_t out_features){
    assert(in_features>0 && out_features>0);
    pg_ag_linear *l = malloc(sizeof(*l));
    if(!l) return NULL;
    size_t wshape[2]={in_features, out_features};
    float limit = sqrtf(6.0f / (float)(in_features + out_features));
    l->weight = pg_var_uniform(2, wshape, -limit, limit, true);
    size_t bshape[1]={out_features};
    l->bias = pg_var_zeros(1, bshape, true);
    if(!l->weight || !l->bias){
        pg_node_free(l->weight); pg_node_free(l->bias); free(l); return NULL;
    }
    l->in_features=in_features; l->out_features=out_features;
    return l;
}
void pg_ag_linear_free(pg_ag_linear *layer){
    if(!layer) return;
    pg_node_free(layer->weight);
    pg_node_free(layer->bias);
    free(layer);
}
pg_node *pg_ag_linear_forward(const pg_ag_linear *layer, pg_node *x){
    assert(layer && x && x->value);
    // y = x @ W + b (broadcast)
    pg_node *mm = pg_ag_matmul(x, layer->weight);
    if(!mm) return NULL;
    pg_node *out = pg_ag_add(mm, layer->bias);
    pg_node_free(mm);
    return out;
}
size_t pg_ag_linear_register(pg_module *m, const pg_ag_linear *l){
    if(!m || !l) return 0;
    pg_module_register(m, l->weight);
    pg_module_register(m, l->bias);
    return 2;
}

/* ---- Sequential ---- */
struct pg_sequential {
    pg_seq_kind_t *kinds;
    void **layers; // for LINEAR points to pg_ag_linear*, for DROPOUT points to float* (p), else NULL
    float *dropout_p;
    size_t n, cap;
};

pg_sequential *pg_sequential_new(void)
{
    pg_sequential *s = calloc(1, sizeof(*s));
    return s;
}

void pg_sequential_free(pg_sequential *seq)
{
    if (!seq)
        return;

    for (size_t i = 0; i < seq->n; i++) {
        if (seq->kinds[i] == PG_SEQ_LINEAR && seq->layers[i])
            pg_ag_linear_free((pg_ag_linear *)seq->layers[i]);
        if (seq->kinds[i] == PG_SEQ_DROPOUT && seq->layers[i])
            free(seq->layers[i]);
    }

    free(seq->kinds);
    free(seq->layers);
    free(seq->dropout_p);
    free(seq);
}

static int seq_grow(pg_sequential *seq)
{
    if (seq->n < seq->cap)
        return 0;

    size_t nc = seq->cap ? seq->cap * 2 : 8;
    pg_seq_kind_t *nk = realloc(seq->kinds, nc * sizeof(*nk));
    void **nl = realloc(seq->layers, nc * sizeof(*nl));
    float *np = realloc(seq->dropout_p, nc * sizeof(*np));

    if (!nk || !nl || !np) {
        free(nk);
        free(nl);
        free(np);
        return -1;
    }

    seq->kinds = nk;
    seq->layers = nl;
    seq->dropout_p = np;
    seq->cap = nc;
    return 0;
}

int pg_sequential_add_linear(pg_sequential *seq, size_t in, size_t out)
{
    if (!seq)
        return -1;
    if (seq_grow(seq) != 0)
        return -1;

    pg_ag_linear *l = pg_ag_linear_new(in, out);
    if (!l)
        return -1;

    seq->kinds[seq->n] = PG_SEQ_LINEAR;
    seq->layers[seq->n] = l;
    seq->dropout_p[seq->n] = 0;
    seq->n++;
    return 0;
}

int pg_sequential_add_activation(pg_sequential *seq, pg_seq_kind_t kind)
{
    if (!seq)
        return -1;
    if (kind == PG_SEQ_LINEAR || kind == PG_SEQ_DROPOUT)
        return -1;
    if (kind != PG_SEQ_RELU && kind != PG_SEQ_TANH && kind != PG_SEQ_SIGMOID &&
        kind != PG_SEQ_GELU && kind != PG_SEQ_LAYERNORM && kind != PG_SEQ_BATCHNORM)
        return -1;
    if (seq_grow(seq) != 0)
        return -1;

    seq->kinds[seq->n] = kind;
    seq->layers[seq->n] = NULL;
    seq->dropout_p[seq->n] = 0;
    seq->n++;
    return 0;
}

int pg_sequential_add_dropout(pg_sequential *seq, float p)
{
    if (!seq || p < 0 || p >= 1)
        return -1;
    if (seq_grow(seq) != 0)
        return -1;

    float *pp = malloc(sizeof(float));
    if (!pp)
        return -1;

    *pp = p;
    seq->kinds[seq->n] = PG_SEQ_DROPOUT;
    seq->layers[seq->n] = pp;
    seq->dropout_p[seq->n] = p;
    seq->n++;
    return 0;
}

pg_node *pg_sequential_forward(pg_sequential *seq, pg_node *x, bool training)
{
    if (!seq || !x)
        return NULL;

    pg_node *cur = pg_node_retain(x);

    for (size_t i = 0; i < seq->n; i++) {
        pg_node *next = NULL;

        switch (seq->kinds[i]) {
        case PG_SEQ_LINEAR: {
            pg_ag_linear *l = (pg_ag_linear *)seq->layers[i];
            next = pg_ag_linear_forward(l, cur);
            break;
        }
        case PG_SEQ_RELU:
            next = pg_ag_relu(cur);
            break;
        case PG_SEQ_TANH:
            next = pg_ag_tanh(cur);
            break;
        case PG_SEQ_SIGMOID:
            next = pg_ag_sigmoid(cur);
            break;
        case PG_SEQ_GELU:
            next = pg_ag_gelu(cur);
            break;
        case PG_SEQ_DROPOUT: {
            float p = seq->dropout_p[i];
            next = pg_ag_dropout(cur, p, training);
            break;
        }
        case PG_SEQ_LAYERNORM:
            next = pg_ag_layernorm(cur, NULL, NULL, 1e-5f);
            break;
        case PG_SEQ_BATCHNORM:
            next = pg_ag_batchnorm(cur, NULL, NULL, 1e-5f);
            break;
        default:
            break;
        }

        pg_node_free(cur);
        if (!next)
            return NULL;
        cur = next;
    }

    return cur;
}

size_t pg_sequential_num_params(pg_sequential *seq)
{
    if (!seq)
        return 0;

    size_t cnt = 0;
    for (size_t i = 0; i < seq->n; i++) {
        if (seq->kinds[i] == PG_SEQ_LINEAR)
            cnt += 2;
    }
    return cnt;
}

int pg_sequential_register(pg_module *m, pg_sequential *seq)
{
    if (!m || !seq)
        return -1;

    for (size_t i = 0; i < seq->n; i++) {
        if (seq->kinds[i] == PG_SEQ_LINEAR) {
            pg_ag_linear *l = (pg_ag_linear *)seq->layers[i];
            if (pg_ag_linear_register(m, l) == 0)
                return -1;
        }
    }
    return 0;
}

size_t pg_sequential_depth(pg_sequential *seq)
{
    return seq ? seq->n : 0;
}
