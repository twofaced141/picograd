#include "nn.h"

#include "../ops/matmul.h"
#include "../ops/elementwise.h"
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

    pg_tensor *y = pg_matmul(x, layer->weight);
    if (!y)
        return NULL;

    pg_tensor *out = pg_add(y, layer->bias);
    pg_tensor_free(y);
    return out;
}
