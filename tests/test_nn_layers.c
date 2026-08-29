#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/tensor.h"
#include "../src/ops/conv.h"
#include "../src/ops/index.h"
#include "../src/autograd/autograd.h"
#include "../src/nn/nn.h"

static int fails = 0;

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            printf("FAIL:%d %s\n", __LINE__, #c);             \
            fails++;                                                   \
        }                                                              \
    } while (0)

static bool closef(float x, float y)
{
    return fabsf(x - y) <= 1e-4f * (1.0f + fabsf(y));
}
/* Looser tolerance for numeric gradient checks: central-diff with eps=1e-2
 * in float32 has ~1e-3 absolute noise. */
static bool closef_grad(float x, float y)
{
    return fabsf(x - y) <= 1e-3f * (1.0f + fabsf(y));
}

static bool close_tensor(const pg_tensor *t, size_t n, const float *ref)
{
    if (!t || t->numel != n)
        return false;
    for (size_t i = 0; i < n; i++)
        if (!closef(t->data[i], ref[i]))
            return false;
    return true;
}

static bool allclose(const pg_tensor *a, const pg_tensor *b, float rtol)
{
    return pg_tensor_allclose(a, b, rtol, rtol);
}

static float *clone_tensor(const pg_tensor *t)
{
    float *d = malloc(t->numel * sizeof(float));
    if (d) memcpy(d, t->data, t->numel * sizeof(float));
    return d;
}

/* ================= Conv2d ================= */

static void test_conv2d_eager_forward(void)
{
    pg_conv2d_cfg cfg = {1, 0};
    size_t shape[4] = {1, 1, 3, 3};
    float data[] = {1,2,3,4,5,6,7,8,9};
    pg_tensor *x = pg_tensor_from_data(4, shape, data);
    size_t wshape[4] = {1,1,1,1};
    float wdata[] = {2.0f};
    pg_tensor *w = pg_tensor_from_data(4, wshape, wdata);
    float bdata[] = {1.0f};
    pg_tensor *b = pg_tensor_from_data(1, (size_t[]){1}, bdata);
    pg_tensor *y = pg_conv2d(x, w, b, cfg);
    CHECK(y && y->numel == 9);
    for (size_t i = 0; i < 9; i++)
        CHECK(closef(y->data[i], data[i] * 2.0f + 1.0f));
    pg_tensor_free(y); pg_tensor_free(b); pg_tensor_free(w); pg_tensor_free(x);
}

static float conv_sum_loss(const pg_tensor *x, const pg_tensor *w,
                              const pg_tensor *b, pg_conv2d_cfg cfg)
{
    pg_tensor *y = pg_conv2d(x, w, b, cfg);
    if (!y) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < y->numel; i++) s += y->data[i];
    pg_tensor_free(y);
    return s;
}

static void test_conv2d_gradients(void)
{
    pg_seed(42);
    size_t N=2, Cin=3, Cout=2, kh=3, kw=3, H=7, W=7;
    pg_conv2d_cfg cfg = {1, 1};
    pg_tensor *x = pg_tensor_uniform(4, (size_t[]){N,Cin,H,W}, -1.0f, 1.0f);
    pg_tensor *w = pg_tensor_uniform(4, (size_t[]){Cout,Cin,kh,kw}, -1.0f, 1.0f);
    pg_tensor *b = pg_tensor_zeros(1, (size_t[]){Cout});
    float eps = 1e-2f;

    pg_node *xn = pg_var_from_tensor(x, true);
    pg_node *wn = pg_var_from_tensor(w, true);
    pg_node *bn = pg_var_from_tensor(b, true);
    pg_node *on = pg_ag_conv2d(xn, wn, bn, kh, kw, cfg.stride, cfg.padding);
    pg_backward(on);
    pg_node_free(on);
    CHECK(xn->grad && wn->grad && bn->grad);

    for (size_t i = 0; i < x->numel; i++) {
        float old = x->data[i];
        x->data[i] = old + eps;
        float l1 = conv_sum_loss(x, w, b, cfg);
        x->data[i] = old - eps;
        float l2 = conv_sum_loss(x, w, b, cfg);
        x->data[i] = old;
        CHECK(closef_grad(xn->grad->data[i], (l1 - l2) / (2.0f * eps)));
    }
    for (size_t i = 0; i < w->numel; i++) {
        float old = w->data[i];
        w->data[i] = old + eps;
        float l1 = conv_sum_loss(x, w, b, cfg);
        w->data[i] = old - eps;
        float l2 = conv_sum_loss(x, w, b, cfg);
        w->data[i] = old;
        CHECK(closef_grad(wn->grad->data[i], (l1 - l2) / (2.0f * eps)));
    }
    for (size_t i = 0; i < b->numel; i++) {
        float old = b->data[i];
        b->data[i] = old + eps;
        float l1 = conv_sum_loss(x, w, b, cfg);
        b->data[i] = old - eps;
        float l2 = conv_sum_loss(x, w, b, cfg);
        b->data[i] = old;
        CHECK(closef(bn->grad->data[i], (l1 - l2) / (2.0f * eps)));
    }
    pg_node_free(xn); pg_node_free(wn); pg_node_free(bn);
    pg_tensor_free(b); pg_tensor_free(w); pg_tensor_free(x);
}

static void test_conv2d_layer(void)
{
    pg_conv2d_layer *l = pg_conv2d_layer_new(3, 2, 3, 3, 1, 1);
    CHECK(l && l->weight && l->bias);
    size_t shape[4] = {2, 3, 7, 7};
    pg_tensor *x = pg_tensor_uniform(4, shape, -1.0f, 1.0f);
    pg_tensor *y = pg_conv2d_layer_forward(l, x);
    CHECK(y && y->ndim == 4 && y->shape[0] == 2 && y->shape[1] == 2);
    pg_node *xn = pg_var_from_tensor(x, true);
    pg_node *on = pg_ag_conv2d_layer_forward(l, xn);
    CHECK(on && on->value);
    pg_node_free(on); pg_node_free(xn); pg_tensor_free(y); pg_tensor_free(x);
    pg_conv2d_layer_free(l);
}

/* ================= Embedding ================= */

static void test_embedding_eager_forward(void)
{
    size_t wshape[2] = {4, 3};
    float wdata[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    pg_tensor *weight = pg_tensor_from_data(2, wshape, wdata);
    float idata[] = {0,2,3,1,0};
    pg_tensor *indices = pg_tensor_from_data(1, (size_t[]){5}, idata);
    pg_tensor *out = pg_embedding(weight, indices);
    CHECK(out && out->numel == 15);
    CHECK(out->shape[0] == 5 && out->shape[1] == 3);
    float ref[] = {1,2,3, 7,8,9, 10,11,12, 4,5,6, 1,2,3};
    CHECK(close_tensor(out, 15, ref));
    pg_tensor_free(out); pg_tensor_free(indices); pg_tensor_free(weight);
}

static float embedding_sum_loss(const pg_tensor *weight, const pg_tensor *indices)
{
    pg_tensor *out = pg_embedding(weight, indices);
    if (!out) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < out->numel; i++) s += out->data[i];
    pg_tensor_free(out);
    return s;
}

static void test_embedding_gradients(void)
{
    pg_seed(99);
    size_t V=6, E=4;
    pg_tensor *weight = pg_tensor_uniform(2, (size_t[]){V,E}, -1.0f, 1.0f);
    float idata[] = {0,3,1,4,2,0,3};
    pg_tensor *indices = pg_tensor_from_data(1, (size_t[]){7}, idata);
    float eps = 1e-2f;

    pg_node *wn = pg_var_from_tensor(weight, true);
    pg_node *on = pg_ag_embedding(wn, indices);
    pg_backward(on);
    pg_node_free(on);
    CHECK(wn->grad);

    for (size_t i = 0; i < weight->numel; i++) {
        float old = weight->data[i];
        weight->data[i] = old + eps;
        float l1 = embedding_sum_loss(weight, indices);
        weight->data[i] = old - eps;
        float l2 = embedding_sum_loss(weight, indices);
        weight->data[i] = old;
        CHECK(closef(wn->grad->data[i], (l1 - l2) / (2.0f * eps)));
    }
    pg_node_free(wn); pg_tensor_free(indices); pg_tensor_free(weight);
}

static void test_embedding_layer(void)
{
    pg_embedding_layer *l = pg_embedding_layer_new(8, 3);
    CHECK(l && l->weight);
    float idata[] = {1,5,2};
    pg_tensor *indices = pg_tensor_from_data(1, (size_t[]){3}, idata);
    pg_tensor *y = pg_embedding_layer_forward(l, indices);
    CHECK(y && y->numel == 9 && y->shape[0] == 3 && y->shape[1] == 3);
    pg_node *yn = pg_ag_embedding_layer_forward(l, indices);
    CHECK(yn && yn->value);
    pg_node_free(yn); pg_tensor_free(y); pg_tensor_free(indices);
    pg_embedding_layer_free(l);
}

/* ================= Dropout ================= */

static void test_dropout_eval_is_identity(void)
{
    float d[] = {1,2,3,4};
    pg_tensor *x = pg_tensor_from_data(1, (size_t[]){4}, d);
    pg_tensor *y = pg_dropout(x, 0.5f, false);
    CHECK(allclose(x, y, 1e-4f));
    pg_tensor_free(y); pg_tensor_free(x);
}

static bool has_zero(const pg_tensor *t)
{
    for (size_t i = 0; i < t->numel; i++)
        if (t->data[i] == 0.0f) return true;
    return false;
}

static void test_dropout_training_forward_backward_consistency(void)
{
    float d[] = {1,2,3,4,5,6};
    pg_tensor *x = pg_tensor_from_data(1, (size_t[]){6}, d);

    /* training: output must be sparse (at least one zero) */
    pg_tensor *y = pg_dropout(x, 0.5f, true);
    CHECK(has_zero(y));
    pg_tensor_free(y);

    /* two independent forward/backward passes on same input must yield
     * different gradients (proves random mask is stored in ctx) */
    float *saved = clone_tensor(x);
    pg_node *xn1 = pg_var_from_tensor(x, true);
    pg_node *on1 = pg_ag_dropout(xn1, 0.5f, true);
    pg_backward(on1);
    float *g1 = clone_tensor(xn1->grad);
    pg_node_free(on1); pg_node_free(xn1);

    memcpy(x->data, saved, x->numel * sizeof(float));
    pg_node *xn2 = pg_var_from_tensor(x, true);
    pg_node *on2 = pg_ag_dropout(xn2, 0.5f, true);
    pg_backward(on2);
    float *g2 = clone_tensor(xn2->grad);
    pg_node_free(on2); pg_node_free(xn2);

    bool same = true;
    for (size_t i = 0; i < x->numel; i++)
        if (g1[i] != g2[i]) { same = false; break; }
    CHECK(!same);

    /* eval path: backward passes grad unchanged */
    pg_node *xe = pg_var_from_tensor(x, true);
    pg_node *oe = pg_ag_dropout(xe, 0.5f, false);
    pg_backward(oe);
    /* eval path: backward passes grad unchanged (identity).
     * grad_out is filled with 1s by pg_backward, so xe->grad == ones. */
    {
        pg_tensor *ones = pg_tensor_full(1, &x->numel, 1.0f);
        CHECK(allclose(xe->grad, ones, 1e-4f));
        pg_tensor_free(ones);
    }
    pg_node_free(oe); pg_node_free(xe);

    free(saved); free(g1); free(g2);
    pg_tensor_free(x);
}

static void test_dropout_layer(void)
{
    pg_dropout_layer *l = pg_dropout_layer_new(0.3f);
    CHECK(l);
    float d[] = {1,2,3,4};
    pg_tensor *x = pg_tensor_from_data(1, (size_t[]){4}, d);
    pg_tensor *y = pg_dropout_layer_forward(l, x, true);
    CHECK(y && y->numel == 4);
    pg_node *tmp_xn = pg_var_from_tensor(x, true);
    pg_node *yn = pg_ag_dropout_layer_forward(tmp_xn, 0.3f, true);
    CHECK(yn && yn->value);
    pg_node_free(tmp_xn);
    pg_node_free(yn); pg_tensor_free(y); pg_tensor_free(x);
    pg_dropout_layer_free(l);
}

int main(void)
{
    test_conv2d_eager_forward();
    test_conv2d_gradients();
    test_conv2d_layer();
    test_embedding_eager_forward();
    test_embedding_gradients();
    test_embedding_layer();
    test_dropout_eval_is_identity();
    test_dropout_training_forward_backward_consistency();
    test_dropout_layer();

    if (fails == 0)
        printf("test_nn_layers: all passed\n");
    else
        printf("test_nn_layers: %d failure(s)\n", fails);
    return fails != 0;
}
