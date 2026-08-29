#include "nn.h"

#include "../ops/conv.h"
#include "../ops/index.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Conv2d ---------- */

pg_conv2d_layer *pg_conv2d_layer_new(size_t in_ch, size_t out_ch,
                                       size_t kh, size_t kw,
                                       int stride, int padding)
{
    assert(in_ch > 0 && out_ch > 0 && kh > 0 && kw > 0);
    pg_conv2d_layer *l = malloc(sizeof(*l));
    if (!l)
        return NULL;
    l->in_channels = in_ch;
    l->out_channels = out_ch;
    l->kernel_h = kh;
    l->kernel_w = kw;
    l->stride = stride;
    l->padding = padding;
    size_t wshape[4] = {out_ch, in_ch, kh, kw};
    float limit = sqrtf(1.0f / (float)(in_ch * kh * kw));
    l->weight = pg_var_uniform(4, wshape, -limit, limit, true);
    size_t bshape[1] = {out_ch};
    l->bias = pg_var_zeros(1, bshape, true);
    if (!l->weight || !l->bias) {
        pg_node_free(l->weight);
        pg_node_free(l->bias);
        free(l);
        return NULL;
    }
    return l;
}

void pg_conv2d_layer_free(pg_conv2d_layer *l)
{
    if (!l)
        return;
    pg_node_free(l->weight);
    pg_node_free(l->bias);
    free(l);
}

pg_tensor *pg_conv2d_layer_forward(const pg_conv2d_layer *l, const pg_tensor *x)
{
    assert(l && x && x->data);
    pg_conv2d_cfg cfg = {l->stride, l->padding};
    return pg_conv2d(x, l->weight->value, l->bias->value, cfg);
}

pg_node *pg_ag_conv2d_layer_forward(const pg_conv2d_layer *l, pg_node *x)
{
    return pg_ag_conv2d(x, l->weight, l->bias, l->kernel_h, l->kernel_w,
                          l->stride, l->padding);
}

size_t pg_conv2d_layer_register(pg_module *m, const pg_conv2d_layer *l)
{
    if (!m || !l)
        return 0;
    pg_module_register(m, l->weight);
    pg_module_register(m, l->bias);
    return 2;
}

/* ---------- Embedding ---------- */

pg_embedding_layer *pg_embedding_layer_new(size_t num_embeddings,
                                                size_t embedding_dim)
{
    assert(num_embeddings > 0 && embedding_dim > 0);
    pg_embedding_layer *l = malloc(sizeof(*l));
    if (!l)
        return NULL;
    l->num_embeddings = num_embeddings;
    l->embedding_dim = embedding_dim;
    size_t wshape[2] = {num_embeddings, embedding_dim};
    float limit = sqrtf(1.0f / (float)embedding_dim);
    l->weight = pg_var_uniform(2, wshape, -limit, limit, true);
    if (!l->weight) {
        free(l);
        return NULL;
    }
    return l;
}

void pg_embedding_layer_free(pg_embedding_layer *l)
{
    if (!l)
        return;
    pg_node_free(l->weight);
    free(l);
}

pg_tensor *pg_embedding_layer_forward(const pg_embedding_layer *l,
                                         const pg_tensor *indices)
{
    assert(l && indices && indices->data);
    return pg_embedding(l->weight->value, indices);
}

pg_node *pg_ag_embedding_layer_forward(const pg_embedding_layer *l,
                                          const pg_tensor *indices)
{
    return pg_ag_embedding(l->weight, indices);
}

size_t pg_embedding_layer_register(pg_module *m, const pg_embedding_layer *l)
{
    if (!m || !l)
        return 0;
    pg_module_register(m, l->weight);
    return 1;
}

/* ---------- Dropout ---------- */

pg_dropout_layer *pg_dropout_layer_new(float p)
{
    assert(p >= 0.0f && p < 1.0f);
    pg_dropout_layer *l = malloc(sizeof(*l));
    if (!l)
        return NULL;
    l->p = p;
    return l;
}

void pg_dropout_layer_free(pg_dropout_layer *l)
{
    free(l);
}

pg_tensor *pg_dropout_layer_forward(const pg_dropout_layer *l,
                                       const pg_tensor *x, bool training)
{
    assert(l && x && x->data);
    return pg_dropout(x, l->p, training);
}

pg_node *pg_ag_dropout_layer_forward(pg_node *x, float p, bool training)
{
    return pg_ag_dropout(x, p, training);
}
