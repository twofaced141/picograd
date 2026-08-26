#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/autograd/autograd.h"
#include "../src/core/tensor.h"

static int fails = 0;

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            printf("FAIL:%d %s\n", __LINE__, #c);                      \
            fails++;                                                   \
        }                                                              \
    } while (0)

#define CHECKF(t, i, v)                                                \
    do {                                                               \
        if (!closef((t)->data[i], (v))) {                              \
            printf("FAIL:%d %s[%zu]=%f want %f\n", __LINE__, #t,       \
                   (size_t)(i), (t)->data[i], (double)(v));            \
            fails++;                                                   \
        }                                                              \
    } while (0)

static bool closef(float x, float y)
{
    return fabsf(x - y) <= 1e-4f * (1.0f + fabsf(y));
}

static bool close_num(float x, float y)
{
    float d = fabsf(x - y);
    return d <= 2e-2f * (1.0f + fabsf(y)) + 1e-3f;
}

typedef pg_node *(*build_fn)(pg_node **xs, void *ud);

static pg_node *owned_sum_all(pg_node *t)
{
    pg_node *l = pg_ag_sum_all(t);
    pg_node_free(t);
    return l;
}

static pg_node *owned_mean_all(pg_node *t)
{
    pg_node *l = pg_ag_mean_all(t);
    pg_node_free(t);
    return l;
}

static void numcheck(const char *name, build_fn build, pg_node **xs,
                     size_t nx, void *ud)
{
    pg_node *loss = build(xs, ud);
    CHECK(loss && loss->value->numel == 1);
    if (!loss)
        return;

    pg_backward(loss);

    pg_tensor *gs[8] = {0};
    for (size_t k = 0; k < nx; k++) {
        CHECK(xs[k]->requires_grad);
        gs[k] = pg_node_grad(xs[k]);
        CHECK(gs[k]);
    }

    const float h = 1e-2f;
    for (size_t k = 0; k < nx; k++) {
        if (!gs[k])
            continue;
        for (size_t i = 0; i < xs[k]->value->numel; i++) {
            float old = xs[k]->value->data[i];

            xs[k]->value->data[i] = old + h;
            pg_node *lp = build(xs, ud);
            float fp = lp ? lp->value->data[0] : NAN;

            xs[k]->value->data[i] = old - h;
            pg_node *lm = build(xs, ud);
            float fm = lm ? lm->value->data[0] : NAN;
            xs[k]->value->data[i] = old;

            float num = (fp - fm) / (2.0f * h);
            if (!close_num(gs[k]->data[i], num)) {
                printf("FAIL:%d %s leaf%zu[%zu] grad=%f numeric=%f\n",
                       __LINE__, name, k, i, gs[k]->data[i], num);
                fails++;
            }
            pg_node_free(lp);
            pg_node_free(lm);
        }
    }
    pg_node_free(loss);
}

static void free_leaves(pg_node **xs, size_t nx)
{
    for (size_t k = 0; k < nx; k++)
        pg_node_free(xs[k]);
}

static void test_lifecycle(void)
{
    float raw[4] = {1, -2, 3, -4};
    size_t shp[1] = {4};
    pg_node *a = pg_var_from_data(1, shp, raw, true);
    CHECK(a && a->refs == 1 && a->requires_grad && !a->grad);
    CHECK(pg_node_value(a)->data[2] == 3.0f);

    pg_node *b = pg_node_retain(a);
    pg_node *c = pg_ag_neg(a);
    CHECK(c && c->nparents == 1 && c->requires_grad);
    CHECKF(pg_node_value(c), 1, 2.0f);

    pg_node_free(a);
    CHECKF(pg_node_value(b), 0, 1.0f);
    pg_node_free(c);
    CHECKF(pg_node_value(b), 2, 3.0f);
    pg_node_free(b);

    pg_node_free(NULL);

    pg_node *ng = pg_var_scalar(5.0f, false);
    pg_node *loss = pg_ag_sum_all(ng);
    pg_backward(loss);
    CHECK(!pg_node_grad(ng));
    CHECK(!pg_node_grad(loss));
    pg_node_free(loss);
    pg_node_free(ng);

    pg_backward(NULL);
}

static pg_node *b_add(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_add(xs[0], xs[1]));
}

static pg_node *b_sub(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_sub(xs[0], xs[1]));
}

static pg_node *b_mul(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_mul(xs[0], xs[1]));
}

static pg_node *b_div(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_div(xs[0], xs[1]));
}

static void test_arithmetic(void)
{
    float xraw[6] = {1.0f, -2.0f, 3.0f, 0.5f, -1.5f, 2.5f};
    float yraw[6] = {2.0f, 0.5f, 4.0f, -1.0f, 3.0f, -2.5f};
    size_t shp[2] = {2, 3};

    pg_node *x = pg_var_from_data(2, shp, xraw, true);
    pg_node *y = pg_var_from_data(2, shp, yraw, true);
    pg_node *xs[2] = {x, y};

    numcheck("add", b_add, xs, 2, NULL);
    numcheck("sub", b_sub, xs, 2, NULL);
    numcheck("mul", b_mul, xs, 2, NULL);
    numcheck("div", b_div, xs, 2, NULL);

    free_leaves(xs, 2);

    float craw[6] = {1, 2, 3, 4, 5, 6};
    pg_node *c = pg_var_from_data(2, shp, craw, true);
    pg_node *z = pg_var_scalar(0.5f, true);

    pg_node *zs[2] = {c, z};
    numcheck("add-bcast-scalar", b_add, zs, 2, NULL);

    float rraw[3] = {1.0f, 2.0f, 3.0f};
    pg_node *r = pg_var_from_data(1, (size_t[]){3}, rraw, true);
    pg_node *rs[2] = {c, r};
    numcheck("mul-bcast-row", b_mul, rs, 2, NULL);

    float colraw[2] = {10.0f, 20.0f};
    pg_node *col = pg_var_from_data(2, (size_t[]){2, 1}, colraw, true);
    pg_node *cs[2] = {col, c};
    numcheck("div-bcast-col", b_div, cs, 2, NULL);

    pg_node_free(c);
    pg_node_free(z);
    pg_node_free(r);
    pg_node_free(col);
}

static pg_node *b_exp(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_exp(xs[0]));
}

static pg_node *b_log(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_log(xs[0]));
}

static pg_node *b_sqrt(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_sqrt(xs[0]));
}

static pg_node *b_sin(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_sin(xs[0]));
}

static pg_node *b_cos(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_cos(xs[0]));
}

static pg_node *b_neg(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_neg(xs[0]));
}

static pg_node *b_relu(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_relu(xs[0]));
}

static pg_node *b_sigmoid(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_sigmoid(xs[0]));
}

static pg_node *b_tanh(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_mean_all(pg_ag_tanh(xs[0]));
}

static void test_unary(void)
{
    float raw[4] = {-1.2f, 0.7f, 1.9f, -0.4f};
    size_t shp[1] = {4};

    struct {
        const char *name;
        build_fn fn;
    } cases[] = {
        {"exp", b_exp},     {"log", b_log},       {"sqrt", b_sqrt},
        {"sin", b_sin},     {"cos", b_cos},       {"neg", b_neg},
        {"relu", b_relu},   {"sigmoid", b_sigmoid}, {"tanh", b_tanh},
    };

    float pos[4] = {0.5f, 1.5f, 2.5f, 4.0f};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        bool positive_domain = cases[i].fn == b_log || cases[i].fn == b_sqrt;
        float *data = positive_domain ? pos : raw;
        pg_node *x = pg_var_from_data(1, shp, data, true);
        pg_node *xs[1] = {x};
        numcheck(cases[i].name, cases[i].fn, xs, 1, NULL);
        free_leaves(xs, 1);
    }
}

static pg_node *b_mm22(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_matmul(xs[0], xs[1]));
}

static void test_matmul(void)
{
    float araw[12] = {0.5f, -1.0f, 2.0f, 0.3f, -0.7f, 1.1f,
                      0.2f, -0.4f, 0.9f, 1.5f, -2.0f, 0.6f};
    float braw[8] = {1.0f, -0.5f, 2.0f, 0.25f, -1.5f, 0.75f, 0.1f, -0.2f};

    pg_node *A = pg_var_from_data(2, (size_t[]){3, 4}, araw, true);
    pg_node *B = pg_var_from_data(2, (size_t[]){4, 2}, braw, true);
    pg_node *ab[2] = {A, B};
    numcheck("matmul-2d2d", b_mm22, ab, 2, NULL);
    free_leaves(ab, 2);

    float vraw[4] = {0.5f, -1.0f, 2.0f, 0.3f};
    pg_node *v = pg_var_from_data(1, (size_t[]){4}, vraw, true);
    pg_node *B2 = pg_var_from_data(2, (size_t[]){4, 2}, braw, true);
    pg_node *vb[2] = {v, B2};
    numcheck("matmul-v2d", b_mm22, vb, 2, NULL);
    free_leaves(vb, 2);

    pg_node *A2 = pg_var_from_data(2, (size_t[]){3, 4}, araw, true);
    pg_node *v2 = pg_var_from_data(1, (size_t[]){4}, vraw, true);
    pg_node *av[2] = {A2, v2};
    numcheck("matmul-2dv", b_mm22, av, 2, NULL);
    free_leaves(av, 2);

    float wraw[4] = {1.0f, 0.5f, -0.5f, 2.0f};
    pg_node *v3 = pg_var_from_data(1, (size_t[]){4}, vraw, true);
    pg_node *w = pg_var_from_data(1, (size_t[]){4}, wraw, true);
    pg_node *vw[2] = {v3, w};
    numcheck("matmul-vv", b_mm22, vw, 2, NULL);
    free_leaves(vw, 2);

    pg_seed(11);
    pg_node *BA = pg_var_uniform(3, (size_t[]){2, 3, 4}, -1.0f, 1.0f, true);
    pg_node *BB = pg_var_uniform(3, (size_t[]){2, 4, 5}, -1.0f, 1.0f, true);
    pg_node *bb[2] = {BA, BB};
    numcheck("matmul-bmm", b_mm22, bb, 2, NULL);
    free_leaves(bb, 2);

    pg_node *BA2 = pg_var_uniform(3, (size_t[]){2, 3, 4}, -1.0f, 1.0f, true);
    pg_node *BW = pg_var_uniform(2, (size_t[]){4, 5}, -1.0f, 1.0f, true);
    pg_node *bw[2] = {BA2, BW};
    numcheck("matmul-bcast-batch", b_mm22, bw, 2, NULL);
    free_leaves(bw, 2);
}

static pg_node *b_sum_a1(pg_node **xs, void *ud)
{
    (void)ud;
    pg_node *s = pg_ag_sum(xs[0], 1, false);
    pg_node *l = pg_ag_sum_all(s);
    pg_node_free(s);
    return l;
}

static pg_node *b_mean_a0k(pg_node **xs, void *ud)
{
    (void)ud;
    pg_node *m = pg_ag_mean(xs[0], 0, true);
    pg_node *l = pg_ag_sum_all(m);
    pg_node_free(m);
    return l;
}

static pg_node *b_mean_a2(pg_node **xs, void *ud)
{
    (void)ud;
    pg_node *m = pg_ag_mean(xs[0], 2, false);
    pg_node *l = pg_ag_sum_all(m);
    pg_node_free(m);
    return l;
}

static void test_reduce(void)
{
    pg_seed(21);
    pg_node *t = pg_var_uniform(3, (size_t[]){2, 3, 4}, -2.0f, 2.0f, true);

    pg_node *ts[1] = {t};
    numcheck("sum-axis1", b_sum_a1, ts, 1, NULL);
    numcheck("mean-axis0-keepdim", b_mean_a0k, ts, 1, NULL);
    numcheck("mean-axis2", b_mean_a2, ts, 1, NULL);
    free_leaves(ts, 1);
}

static pg_node *b_softmax(pg_node **xs, void *ud)
{
    pg_node *sm = pg_ag_softmax(xs[0], 1);
    pg_node *m = pg_ag_mul(sm, ud);
    pg_node_free(sm);
    return owned_sum_all(m);
}

static void test_softmax(void)
{
    pg_seed(33);
    pg_node *x = pg_var_uniform(2, (size_t[]){3, 4}, -2.0f, 2.0f, true);
    pg_node *c = pg_var_uniform(2, (size_t[]){3, 4}, -1.0f, 1.0f, false);

    pg_node *xs[1] = {x};
    numcheck("softmax-axis1", b_softmax, xs, 1, c);
    free_leaves(xs, 1);
    pg_node_free(c);
}

static pg_node *b_mlp(pg_node **xs, void *ud)
{
    (void)ud;
    pg_node *mm1 = pg_ag_matmul(xs[0], xs[1]);
    pg_node *z1 = pg_ag_add(mm1, xs[2]);
    pg_node_free(mm1);
    pg_node *h = pg_ag_sigmoid(z1);
    pg_node_free(z1);
    pg_node *mm2 = pg_ag_matmul(h, xs[3]);
    pg_node_free(h);
    pg_node *y = pg_ag_add(mm2, xs[4]);
    pg_node_free(mm2);
    return owned_mean_all(y);
}

static void test_mlp(void)
{
    pg_seed(55);
    pg_node *x = pg_var_uniform(2, (size_t[]){2, 3}, -1.0f, 1.0f, true);
    pg_node *w1 = pg_var_uniform(2, (size_t[]){3, 8}, -0.5f, 0.5f, true);
    pg_node *b1 = pg_var_zeros(1, (size_t[]){8}, true);
    pg_node *w2 = pg_var_uniform(2, (size_t[]){8, 2}, -0.5f, 0.5f, true);
    pg_node *b2 = pg_var_zeros(1, (size_t[]){2}, true);

    pg_node *xs[5] = {x, w1, b1, w2, b2};
    numcheck("mlp", b_mlp, xs, 5, NULL);

    free_leaves(xs, 5);
}

static pg_node *b_quad(pg_node **xs, void *ud)
{
    (void)ud;
    return owned_sum_all(pg_ag_mul(xs[0], xs[0]));
}

static pg_node *b_mixed_reuse(pg_node **xs, void *ud)
{
    (void)ud;
    pg_node *x = xs[0];
    pg_node *sq = pg_ag_mul(x, x);
    pg_node *s = pg_ag_add(sq, x);
    pg_node_free(sq);
    return owned_sum_all(s);
}

static void test_analytic(void)
{
    float raw[4] = {0.5f, -1.0f, 2.0f, -0.25f};
    size_t shp[1] = {4};

    pg_node *x = pg_var_from_data(1, shp, raw, true);
    pg_node *loss = b_quad(&x, NULL);
    pg_backward(loss);
    pg_tensor *g = pg_node_grad(x);
    CHECK(g);
    for (size_t i = 0; i < 4; i++)
        CHECKF(g, i, 2.0f * raw[i]);
    pg_node_free(loss);
    pg_node_free(x);

    pg_node *y = pg_var_from_data(1, shp, raw, true);
    pg_node *loss2 = b_mixed_reuse(&y, NULL);
    pg_backward(loss2);
    pg_tensor *g2 = pg_node_grad(y);
    CHECK(g2);
    for (size_t i = 0; i < 4; i++)
        CHECKF(g2, i, 2.0f * raw[i] + 1.0f);

    pg_backward(loss2);
    pg_tensor *g3 = pg_node_grad(y);
    CHECK(g3 == g2);
    for (size_t i = 0; i < 4; i++)
        CHECKF(g3, i, 2.0f * raw[i] + 1.0f);

    pg_node_free(loss2);
    pg_node_free(y);
}

int main(void)
{
    test_lifecycle();
    test_arithmetic();
    test_unary();
    test_matmul();
    test_reduce();
    test_softmax();
    test_mlp();
    test_analytic();

    if (fails == 0)
        printf("test_autograd: all passed\n");
    else
        printf("test_autograd: %d failures\n", fails);
    return fails != 0;
}
