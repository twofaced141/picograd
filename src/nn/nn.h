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

/* ---- Autograd Linear (pg_node based) ---- */
typedef struct {
    size_t in_features;
    size_t out_features;
    pg_node *weight; // [in, out]
    pg_node *bias;   // [out]
} pg_ag_linear;

pg_ag_linear *pg_ag_linear_new(size_t in_features, size_t out_features);
void pg_ag_linear_free(pg_ag_linear *layer);
pg_node *pg_ag_linear_forward(const pg_ag_linear *layer, pg_node *x);
size_t pg_ag_linear_register(pg_module *m, const pg_ag_linear *l);

/* ---- Sequential (autograd) ---- */
typedef enum {
    PG_SEQ_LINEAR = 0,
    PG_SEQ_RELU,
    PG_SEQ_TANH,
    PG_SEQ_SIGMOID,
    PG_SEQ_GELU,
    PG_SEQ_DROPOUT,
    PG_SEQ_LAYERNORM,
    PG_SEQ_BATCHNORM
} pg_seq_kind_t;

typedef struct pg_sequential pg_sequential;

pg_sequential *pg_sequential_new(void);
void pg_sequential_free(pg_sequential *seq);
int pg_sequential_add_linear(pg_sequential *seq, size_t in, size_t out);
int pg_sequential_add_activation(pg_sequential *seq, pg_seq_kind_t kind);
int pg_sequential_add_dropout(pg_sequential *seq, float p);
pg_node *pg_sequential_forward(pg_sequential *seq, pg_node *x, bool training);
size_t pg_sequential_num_params(pg_sequential *seq);
int pg_sequential_register(pg_module *m, pg_sequential *seq);
size_t pg_sequential_depth(pg_sequential *seq);

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
