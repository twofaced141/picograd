#include <stdio.h>

#include "../src/autograd/autograd.h"
#include "../src/opt/sgd.h"

#define EPOCHS 3000
#define PRINT_EVERY 1000

static pg_node *forward(pg_node *x, pg_node *w1, pg_node *b1,
                        pg_node *w2, pg_node *b2)
{
    pg_node *mm1 = pg_ag_matmul(x, w1);
    pg_node *z1 = pg_ag_add(mm1, b1);
    pg_node_free(mm1);

    pg_node *h = pg_ag_tanh(z1);
    pg_node_free(z1);

    pg_node *mm2 = pg_ag_matmul(h, w2);
    pg_node_free(h);

    pg_node *y = pg_ag_add(mm2, b2);
    pg_node_free(mm2);
    return y;
}

static pg_node *mse_loss(pg_node *y, pg_node *target)
{
    pg_node *d = pg_ag_sub(y, target);
    pg_node *sq = pg_ag_mul(d, d);
    pg_node_free(d);
    pg_node *loss = pg_ag_mean_all(sq);
    pg_node_free(sq);
    return loss;
}

int main(void)
{
    pg_seed(7);

    float xraw[8] = {0, 0, 0, 1, 1, 0, 1, 1};
    float yraw[4] = {0, 1, 1, 0};

    pg_node *x = pg_var_from_data(2, (size_t[]){4, 2}, xraw, false);
    pg_node *y = pg_var_from_data(2, (size_t[]){4, 1}, yraw, false);

    pg_node *w1 = pg_var_uniform(2, (size_t[]){2, 8}, -1.0f, 1.0f, true);
    pg_node *b1 = pg_var_zeros(1, (size_t[]){8}, true);
    pg_node *w2 = pg_var_uniform(2, (size_t[]){8, 1}, -1.0f, 1.0f, true);
    pg_node *b2 = pg_var_zeros(1, (size_t[]){1}, true);

    if (!x || !y || !w1 || !b1 || !w2 || !b2)
        return 1;

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.1f;
    cfg.momentum = 0.9f;

    pg_sgd *opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt, w1);
    pg_sgd_add_param(opt, b1);
    pg_sgd_add_param(opt, w2);
    pg_sgd_add_param(opt, b2);

    for (int it = 0; it < EPOCHS; it++) {
        pg_node *out = forward(x, w1, b1, w2, b2);
        pg_node *loss = mse_loss(out, y);
        pg_node_free(out);

        pg_backward(loss);

        float lv = pg_node_value(loss)->data[0];
        if (it % PRINT_EVERY == 0 || it == EPOCHS - 1)
            printf("iter %5d  loss %.6f\n", it, lv);

        pg_sgd_step(opt);
        pg_node_free(loss);
    }

    printf("\npredictions:\n");
    pg_node *out = forward(x, w1, b1, w2, b2);
    int correct = 0;
    const float targets[4] = {0, 1, 1, 0};
    for (size_t i = 0; i < 4; i++) {
        float p = pg_node_value(out)->data[i];
        int cls = p > 0.5f;
        correct += cls == (int)targets[i];
        printf("  (%.0f, %.0f) -> %.4f\n", xraw[i * 2], xraw[i * 2 + 1], p);
    }
    pg_node_free(out);

    printf("accuracy %d/4\n", correct);

    pg_sgd_free(opt);
    pg_node_free(w1);
    pg_node_free(b1);
    pg_node_free(w2);
    pg_node_free(b2);
    pg_node_free(x);
    pg_node_free(y);

    return correct != 4;
}
