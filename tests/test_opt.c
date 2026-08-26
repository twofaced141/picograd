#include <math.h>
#include <stdio.h>

#include "../src/autograd/autograd.h"
#include "../src/opt/sgd.h"

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
    return fabsf(x - y) <= 1e-5f * (1.0f + fabsf(y));
}

/* loss = (x - target)^2 summed; grad = 2 * (x - target) */
static float sgd_step_plain(pg_node *x, float target)
{
    pg_node *t = pg_var_scalar(target, false);
    pg_node *diff = pg_ag_sub(x, t);
    pg_node *sq = pg_ag_mul(diff, diff);
    pg_node *loss = pg_ag_sum_all(sq);
    pg_backward(loss);
    pg_node_free(loss);
    pg_node_free(sq);
    pg_node_free(diff);
    pg_node_free(t);
    return 2.0f * (pg_node_value(x)->data[0] - target);
}

static void test_sgd_vanilla(void)
{
    float d[] = {1.0f};
    pg_node *x = pg_var_from_data(1, (size_t[]){1}, d, true);

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.1f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    CHECK(pg_sgd_add_param(opt, x) == 0);
    CHECK(pg_sgd_num_params(opt) == 1);

    /* step 1: g = -2 -> x = 1 + 0.1*2 = 1.2 */
    pg_sgd_zero_grad(opt);
    float g = sgd_step_plain(x, 2.0f);
    CHECK(closef(g, -2.0f));
    pg_sgd_step(opt);
    CHECK(closef(x->value->data[0], 1.2f));

    /* step 2: g = -1.6 -> x = 1.2 + 0.16 = 1.36 */
    pg_sgd_zero_grad(opt);
    g = sgd_step_plain(x, 2.0f);
    CHECK(closef(g, -1.6f));
    pg_sgd_step(opt);
    CHECK(closef(x->value->data[0], 1.36f));

    pg_sgd_free(opt); /* retains ownership of params */
    pg_node_free(x);
}

static void test_sgd_momentum(void)
{
    float d[] = {1.0f};
    pg_node *x = pg_var_from_data(1, (size_t[]){1}, d, true);

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.1f;
    cfg.momentum = 0.9f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    CHECK(pg_sgd_add_param(opt, x) == 0);

    /* step 1: buf = -2 -> x = 1.2 */
    pg_sgd_zero_grad(opt);
    sgd_step_plain(x, 2.0f);
    pg_sgd_step(opt);
    CHECK(closef(x->value->data[0], 1.2f));

    /* step 2: buf = 0.9*(-2) - 1.6 = -3.4 -> x = 1.54 */
    pg_sgd_zero_grad(opt);
    sgd_step_plain(x, 2.0f);
    pg_sgd_step(opt);
    CHECK(closef(x->value->data[0], 1.54f));

    pg_sgd_free(opt);
    pg_node_free(x);
}

static void test_sgd_weight_decay_nesterov(void)
{
    /* loss = x^2 (single element): grad = 2*x
     * wd = 0.5, lr = 0.1: update = 2 + 0.5 -> x = 1 - 0.25 = 0.75 */
    float d[] = {1.0f};
    pg_node *x = pg_var_from_data(1, (size_t[]){1}, d, true);

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.1f;
    cfg.weight_decay = 0.5f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt, x);

    pg_sgd_zero_grad(opt);
    pg_node *sq = pg_ag_mul(x, x);
    pg_node *loss = pg_ag_sum_all(sq);
    pg_backward(loss);
    pg_node_free(loss);
    pg_node_free(sq);
    pg_sgd_step(opt);
    CHECK(closef(x->value->data[0], 0.75f));

    pg_sgd_free(opt);
    pg_node_free(x);

    /* nesterov, mu = 0.5, lr = 0.1:
     * step1: buf = -2, update = -2 + 0.5*(-2) = -3 -> x = 1.3
     * step2: g = 2*(1.3 - 2) = -1.4, buf = -2.4,
     *        update = -1.4 + 0.5*(-2.4) = -2.6 -> x = 1.56 */
    pg_node *y = pg_var_from_data(1, (size_t[]){1}, d, true);
    cfg = pg_sgd_cfg_default();
    cfg.lr = 0.1f;
    cfg.momentum = 0.5f;
    cfg.nesterov = true;
    opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt, y);

    pg_sgd_zero_grad(opt);
    sgd_step_plain(y, 2.0f);
    pg_sgd_step(opt);
    CHECK(closef(y->value->data[0], 1.3f));

    pg_sgd_zero_grad(opt);
    sgd_step_plain(y, 2.0f);
    pg_sgd_step(opt);
    CHECK(closef(y->value->data[0], 1.56f));

    pg_sgd_free(opt);
    pg_node_free(y);
}

int main(void)
{
    test_sgd_vanilla();
    test_sgd_momentum();
    test_sgd_weight_decay_nesterov();

    if (fails)
        printf("test_opt: %d failure(s)\n", fails);
    else
        printf("test_opt: all passed\n");
    return fails ? 1 : 0;
}
