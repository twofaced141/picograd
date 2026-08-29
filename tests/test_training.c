#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/autograd/autograd.h"
#include "../src/opt/adam.h"
#include "../src/opt/lr_sched.h"
#include "../src/nn/module.h"

static int fails = 0;

#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            printf("FAIL:%d %s\n", __LINE__, #c);                             \
            fails++;                                                          \
        }                                                                     \
    } while (0)

static bool closef(float x, float y)
{
    return fabsf(x - y) <= 1e-3f * (1.0f + fabsf(y));
}

static float ce_eval(pg_node *logits, const size_t *targets, size_t n, bool mean,
                     const float *data, size_t numel)
{
    memcpy(logits->value->data, data, numel * sizeof(float));
    pg_node *loss = pg_ag_cross_entropy(logits, targets, n, mean);
    float v = loss->value->data[0];
    pg_node_free(loss);
    return v;
}

static void test_cross_entropy(void)
{
    size_t rows = 3, cls = 5;
    size_t n = rows * cls;
    float *d = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        d[i] = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;
    size_t targets[3] = {2, 0, 4};

    pg_node *logits = pg_var_from_data(2, (size_t[]){rows, cls}, d, true);

    // forward vs manual stable reference
    pg_node *loss = pg_ag_cross_entropy(logits, targets, rows, true);
    float ref = 0.0f;
    for (size_t r = 0; r < rows; r++) {
        size_t base = r * cls;
        float m = -INFINITY;
        for (size_t c = 0; c < cls; c++)
            if (d[base + c] > m)
                m = d[base + c];
        float s = 0.0f;
        for (size_t c = 0; c < cls; c++)
            s += expf(d[base + c] - m);
        ref += (logf(s) + m) - d[base + targets[r]];
    }
    ref /= (float)rows;
    CHECK(closef(loss->value->data[0], ref));
    pg_node_free(loss);

    // numeric gradient check (sum reduction and mean reduction)
    for (int mean = 0; mean < 2; mean++) {
        float *base = malloc(n * sizeof(float));
        memcpy(base, d, n * sizeof(float));
        float eps = 1e-2f;
        for (size_t i = 0; i < n; i++) {
            float saved = base[i];
            base[i] = saved + eps;
            float fp = ce_eval(logits, targets, rows, mean != 0, base, n);
            base[i] = saved - eps;
            float fm = ce_eval(logits, targets, rows, mean != 0, base, n);
            base[i] = saved;
            float ng = (fp - fm) / (2.0f * eps);

            memcpy(logits->value->data, base, n * sizeof(float));
            pg_node *l = pg_ag_cross_entropy(logits, targets, rows, mean != 0);
            pg_backward(l);
            float ag = logits->grad->data[i];
            pg_node_free(l);
            CHECK(closef(ag, ng));
        }
        free(base);
    }

    // invariant: mean CE grad rows sum to zero (softmax - onehot)
    memcpy(logits->value->data, d, n * sizeof(float));
    pg_node *l2 = pg_ag_cross_entropy(logits, targets, rows, true);
    pg_backward(l2);
    for (size_t r = 0; r < rows; r++) {
        float s = 0.0f;
        for (size_t c = 0; c < cls; c++)
            s += logits->grad->data[r * cls + c];
        CHECK(closef(s, 0.0f));
    }
    pg_node_free(l2);

    pg_node_free(logits);
    free(d);
}

static void test_mse(void)
{
    size_t n = 7;
    float pred[7], tgt[7];
    for (size_t i = 0; i < n; i++) {
        pred[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        tgt[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
    }

    pg_node *p = pg_var_from_data(1, (size_t[]){n}, pred, true);
    pg_node *t = pg_var_from_data(1, (size_t[]){n}, tgt, false);

    float ref = 0.0f;
    for (size_t i = 0; i < n; i++)
        ref += (pred[i] - tgt[i]) * (pred[i] - tgt[i]);
    ref /= (float)n;

    pg_node *loss = pg_ag_mse(p, t, true);
    CHECK(closef(loss->value->data[0], ref));

    // numeric grad
    float *base = malloc(n * sizeof(float));
    memcpy(base, pred, n * sizeof(float));
    float eps = 1e-2f;
    for (size_t i = 0; i < n; i++) {
        float saved = base[i];
        base[i] = saved + eps;
        memcpy(p->value->data, base, n * sizeof(float));
        pg_node *l1 = pg_ag_mse(p, t, true);
        float fp = l1->value->data[0];
        pg_node_free(l1);
        base[i] = saved - eps;
        memcpy(p->value->data, base, n * sizeof(float));
        pg_node *l2 = pg_ag_mse(p, t, true);
        float fm = l2->value->data[0];
        pg_node_free(l2);
        base[i] = saved;
        float ng = (fp - fm) / (2.0f * eps);

        memcpy(p->value->data, base, n * sizeof(float));
        pg_node *l3 = pg_ag_mse(p, t, true);
        pg_backward(l3);
        float ag = p->grad->data[i];
        pg_node_free(l3);
        CHECK(closef(ag, ng));
    }
    free(base);

    pg_node_free(loss);
    pg_node_free(t);
    pg_node_free(p);
}

static void test_bce(void)
{
    size_t n = 6;
    float z[6], t[6];
    for (size_t i = 0; i < n; i++) {
        z[i] = ((float)rand() / RAND_MAX - 0.5f) * 8.0f;
        t[i] = (float)(rand() % 2);
    }

    pg_node *logits = pg_var_from_data(1, (size_t[]){n}, z, true);

    float ref = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float zi = z[i], ti = t[i];
        ref += fmaxf(zi, 0.0f) - zi * ti + log1pf(expf(-fabsf(zi)));
    }

    pg_node *loss = pg_ag_bce_with_logits(logits, t, n, false);
    CHECK(closef(loss->value->data[0], ref));

    float *base = malloc(n * sizeof(float));
    memcpy(base, z, n * sizeof(float));
    float eps = 1e-2f;
    for (size_t i = 0; i < n; i++) {
        float saved = base[i];
        base[i] = saved + eps;
        memcpy(logits->value->data, base, n * sizeof(float));
        pg_node *l1 = pg_ag_bce_with_logits(logits, t, n, false);
        float fp = l1->value->data[0];
        pg_node_free(l1);
        base[i] = saved - eps;
        memcpy(logits->value->data, base, n * sizeof(float));
        pg_node *l2 = pg_ag_bce_with_logits(logits, t, n, false);
        float fm = l2->value->data[0];
        pg_node_free(l2);
        base[i] = saved;
        float ng = (fp - fm) / (2.0f * eps);

        memcpy(logits->value->data, base, n * sizeof(float));
        pg_node *l3 = pg_ag_bce_with_logits(logits, t, n, false);
        pg_backward(l3);
        float ag = logits->grad->data[i];
        pg_node_free(l3);
        CHECK(closef(ag, ng));
    }
    free(base);

    pg_node_free(loss);
    pg_node_free(logits);
}

static void test_lr_sched(void)
{
    // cosine, no warmup: step0 = base, last step ~= min
    pg_lr_sched *c = pg_lr_sched_new_cosine(0.1f, 0.001f, 0, 1000);
    CHECK(closef(pg_lr_sched_current(c), 0.1f));
    for (size_t i = 1; i < 1000; i++)
        pg_lr_sched_step(c);
    CHECK(closef(pg_lr_sched_current(c), 0.001f));
    pg_lr_sched_free(c);

    // linear decay: step0 = base + (min-base)*(1/total) = 0.09, end ~= min
    pg_lr_sched *l = pg_lr_sched_new_linear(0.1f, 0.0f, 10);
    CHECK(closef(pg_lr_sched_current(l), 0.09f));
    for (size_t i = 1; i < 10; i++)
        pg_lr_sched_step(l);
    CHECK(closef(pg_lr_sched_current(l), 0.0f));
    pg_lr_sched_free(l);

    // step decay
    pg_lr_sched *s = pg_lr_sched_new_step(0.1f, 0.5f, 3);
    CHECK(closef(pg_lr_sched_current(s), 0.1f));
    pg_lr_sched_step(s);
    pg_lr_sched_step(s);
    pg_lr_sched_step(s);
    CHECK(closef(pg_lr_sched_current(s), 0.05f)); // 3rd step -> *0.5
    pg_lr_sched_free(s);
}

static void test_module_and_groups(void)
{
    float a[1] = {1.0f}, b[1] = {1.0f};
    pg_node *x = pg_var_from_data(1, (size_t[]){1}, a, true);
    pg_node *y = pg_var_from_data(1, (size_t[]){1}, b, true);

    pg_module *m = pg_module_new();
    pg_module_register(m, x);
    pg_module_register(m, y);
    CHECK(pg_module_num_params(m) == 2);

    pg_adam_cfg cfg = pg_adam_cfg_default();
    cfg.lr = 0.1f;
    pg_adam *opt = pg_adam_new(&cfg);
    // x with small lr override, y with default
    pg_module_add_to_adam_lr(m, opt, 0.0f, -1.0f); // all default
    pg_adam_set_param_lr(opt, 0, 0.01f);           // x gets 0.01
    CHECK(closef(pg_adam_get_lr(opt), 0.1f));

    // give both grad = 1, step once
    x->grad = pg_tensor_full(1, (size_t[]){1}, 1.0f);
    y->grad = pg_tensor_full(1, (size_t[]){1}, 1.0f);
    pg_adam_step(opt);

    float dx = 1.0f - x->value->data[0];
    float dy = 1.0f - y->value->data[0];
    CHECK(dy > dx); // y (lr 0.1) moved more than x (lr 0.01)

    pg_adam_free(opt);
    pg_module_free(m);
    pg_node_free(x);
    pg_node_free(y);
}

int main(void)
{
    test_cross_entropy();
    test_mse();
    test_bce();
    test_lr_sched();
    test_module_and_groups();

    if (fails)
        printf("test_training: %d failure(s)\n", fails);
    else
        printf("test_training: all passed\n");
    return fails ? 1 : 0;
}
