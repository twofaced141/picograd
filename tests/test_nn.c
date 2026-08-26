#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/core/tensor.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/reduce.h"
#include "../src/ops/activations.h"
#include "../src/nn/nn.h"

static int fails = 0;

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            printf("FAIL:%d %s\n", __LINE__, #c);                      \
            fails++;                                                   \
        }                                                              \
    } while (0)

static bool closef(float x, float y)
{
    return fabsf(x - y) <= 1e-4f * (1.0f + fabsf(y));
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

static void test_elementwise(void)
{
    float d[] = {1.0f, -2.0f, 4.0f, 9.0f};
    pg_tensor *a = pg_tensor_from_data(1, (size_t[]){4}, d);

    {
        pg_tensor *r = pg_sub(a, a);
        float ref[4] = {0, 0, 0, 0};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_neg(a);
        float ref[4] = {-1, 2, -4, -9};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_abs(a);
        float ref[4] = {1, 2, 4, 9};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *pos = pg_abs(a);
        pg_tensor *r = pg_sqrt(pos);
        float ref[4] = {1, sqrtf(2.0f), 2, 3};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(pos);
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_clamp(a, -1.5f, 5.0f);
        float ref[4] = {1, -1.5f, 4, 5};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *b = pg_tensor_full(1, (size_t[]){1}, 2.0f);
        pg_tensor *r = pg_sub(a, b);
        float ref[4] = {-1, -4, 2, 7};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(b);
        pg_tensor_free(r);
    }

    pg_tensor_free(a);
}

static void test_activations(void)
{
    float d[] = {-1.0f, 0.0f, 0.5f, 2.0f};
    pg_tensor *a = pg_tensor_from_data(1, (size_t[]){4}, d);

    {
        pg_tensor *r = pg_relu(a);
        float ref[4] = {0, 0, 0.5f, 2};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_leaky_relu(a, 0.1f);
        float ref[4] = {-0.1f, 0, 0.5f, 2};
        CHECK(close_tensor(r, 4, ref));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_sigmoid(a);
        for (size_t i = 0; i < 4; i++)
            CHECK(closef(r->data[i], 1.0f / (1.0f + expf(-d[i]))));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_tanh(a);
        for (size_t i = 0; i < 4; i++)
            CHECK(closef(r->data[i], tanhf(d[i])));
        pg_tensor_free(r);
    }
    {
        pg_tensor *r = pg_gelu(a);
        for (size_t i = 0; i < 4; i++)
            CHECK(closef(r->data[i],
                         0.5f * d[i] * (1.0f + erff(d[i] * 0.70710678118f))));
        pg_tensor_free(r);
    }

    pg_tensor_free(a);

    pg_seed(42);
    pg_tensor *x = pg_tensor_uniform(2, (size_t[]){3, 7}, -4.0f, 4.0f);

    {
        pg_tensor *s = pg_softmax(x, 1);
        pg_tensor *sum = pg_sum(s, 1, false);
        for (size_t i = 0; i < 3; i++)
            CHECK(closef(sum->data[i], 1.0f));

        for (size_t b = 0; b < 3; b++) {
            float m = -INFINITY, z = 0.0f;
            for (size_t j = 0; j < 7; j++) {
                float v = x->data[b * 7 + j];
                if (v > m)
                    m = v;
            }
            for (size_t j = 0; j < 7; j++)
                z += expf(x->data[b * 7 + j] - m);
            for (size_t j = 0; j < 7; j++)
                CHECK(closef(s->data[b * 7 + j],
                             expf(x->data[b * 7 + j] - m) / z));
        }
        pg_tensor_free(s);
        pg_tensor_free(sum);
    }
    {
        pg_tensor *ls = pg_log_softmax(x, 0);
        for (size_t j = 0; j < 7; j++) {
            float m = -INFINITY, z = 0.0f;
            for (size_t b = 0; b < 3; b++) {
                float v = x->data[b * 7 + j];
                if (v > m)
                    m = v;
            }
            for (size_t b = 0; b < 3; b++)
                z += expf(x->data[b * 7 + j] - m);
            for (size_t b = 0; b < 3; b++)
                CHECK(closef(expf(ls->data[b * 7 + j]),
                             expf(x->data[b * 7 + j] - m) / z));
        }
        pg_tensor_free(ls);
    }

    pg_tensor_free(x);
}

static void test_linear(void)
{
    pg_seed(7);
    pg_linear *l = pg_linear_new(4, 3);
    CHECK(l && l->weight && l->bias);

    pg_tensor *x = pg_tensor_arange(0.0f, 10.0f, 0.5f);
    CHECK(pg_tensor_reshape(x, 2, (size_t[]){5, 4}));

    pg_tensor *y = pg_linear_forward(l, x);
    CHECK(y && pg_shape_equal(y->ndim, y->shape, 2, (size_t[]){5, 3}));

    for (size_t b = 0; b < 5; b++) {
        for (size_t o = 0; o < 3; o++) {
            float acc = l->bias->data[o];
            for (size_t i = 0; i < 4; i++)
                acc += x->data[b * 4 + i] * l->weight->data[i * 3 + o];
            CHECK(closef(y->data[b * 3 + o], acc));
        }
    }

    pg_tensor_free(y);
    pg_tensor_free(x);
    pg_linear_free(l);
}

int main(void)
{
    test_elementwise();
    test_activations();
    test_linear();

    if (fails == 0)
        printf("test_nn: all passed\n");
    else
        printf("test_nn: %d failures\n", fails);
    return fails != 0;
}
