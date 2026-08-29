#ifndef PICOGRAD_NN_NN_H
#define PICOGRAD_NN_NN_H

#include "../core/tensor.h"
#include "../autograd/autograd.h"
#include "module.h"

typedef struct {
    size_t in_features;
    size_t out_features;
    pg_tensor *weight;
    pg_tensor *bias;
} pg_linear;

pg_linear *pg_linear_new(size_t in_features, size_t out_features);
void pg_linear_free(pg_linear *layer);
pg_tensor *pg_linear_forward(const pg_linear *layer, const pg_tensor *x);
pg_tensor *pg_linear_forward_relu(const pg_linear *layer, const pg_tensor *x);

/* ---- Conv2d (NCHW). Parameters stored as autograd leaf nodes. ---- */
typedef struct {
    size_t in_channels, out_channels;
    size_t kernel_h, kernel_w;
    int stride, padding;
    pg_node *weight;
    pg_node *bias;
} pg_conv2d_layer;

pg_conv2d_layer *pg_conv2d_layer_new(size_t in_ch, size_t out_ch,
                                       size_t kh, size_t kw,
                                       int stride, int padding);
void pg_conv2d_layer_free(pg_conv2d_layer *l);
pg_tensor *pg_conv2d_layer_forward(const pg_conv2d_layer *l, const pg_tensor *x);
pg_node *pg_ag_conv2d_layer_forward(const pg_conv2d_layer *l, pg_node *x);
size_t pg_conv2d_layer_register(pg_module *m, const pg_conv2d_layer *l);

/* ---- Embedding lookup. Parameters stored as autograd leaf nodes. ---- */
typedef struct {
    size_t num_embeddings, embedding_dim;
    pg_node *weight;
} pg_embedding_layer;

pg_embedding_layer *pg_embedding_layer_new(size_t num_embeddings,
                                            size_t embedding_dim);
void pg_embedding_layer_free(pg_embedding_layer *l);
pg_tensor *pg_embedding_layer_forward(const pg_embedding_layer *l,
                                        const pg_tensor *indices);
pg_node *pg_ag_embedding_layer_forward(const pg_embedding_layer *l,
                                         const pg_tensor *indices);
size_t pg_embedding_layer_register(pg_module *m, const pg_embedding_layer *l);

/* Eager dropout (operates on plain tensors). See pg_ag_dropout in autograd.h. */
pg_tensor *pg_dropout(const pg_tensor *x, float p, bool training);

/* ---- Dropout (no parameters). ---- */
typedef struct { float p; } pg_dropout_layer;

pg_dropout_layer *pg_dropout_layer_new(float p);
void pg_dropout_layer_free(pg_dropout_layer *l);
pg_tensor *pg_dropout_layer_forward(const pg_dropout_layer *l,
                                    const pg_tensor *x, bool training);
pg_node *pg_ag_dropout_layer_forward(pg_node *x, float p, bool training);

#endif
