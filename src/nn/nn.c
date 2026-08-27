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
