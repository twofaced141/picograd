/* Toy end-to-end training demo for picograd Phase 1 infra:
 *   - a 2-hidden-layer MLP on a 3-class spiral
 *   - cross-entropy loss, AdamW, cosine LR schedule, module registry
 * Build/run: make build/train_toy && ./build/train_toy
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/autograd/autograd.h"
#include "../src/nn/module.h"
#include "../src/opt/adam.h"
#include "../src/opt/lr_sched.h"

#define N 300
#define HID 16
#define CLS 3
#define ITERS 2000

static void make_spiral(float *X, size_t *y)
{
    pg_seed(0);
    size_t per = N / CLS;
    for (size_t i = 0; i < N; i++) {
        size_t cls = i / per;
        float r = (float)(i % per) / (float)per * 0.8f + 0.1f;
        float t = cls * 2.094395102f + r * 4.0f;
        X[i * 2 + 0] = r * sinf(t) + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        X[i * 2 + 1] = r * cosf(t) + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        y[i] = cls;
    }
}

int main(void)
{
    float X[N * 2];
    size_t y[N];
    make_spiral(X, y);

    // parameters (param nodes)
    pg_node *W1 = pg_var_uniform(2, (size_t[]){2, HID}, -1.0f, 1.0f, true);
    pg_node *b1 = pg_var_zeros(1, (size_t[]){HID}, true);
    pg_node *W2 = pg_var_uniform(2, (size_t[]){HID, CLS}, -1.0f, 1.0f, true);
    pg_node *b2 = pg_var_zeros(1, (size_t[]){CLS}, true);

    pg_module *m = pg_module_new();
    pg_module_register(m, W1);
    pg_module_register(m, b1);
    pg_module_register(m, W2);
    pg_module_register(m, b2);

    pg_adam_cfg cfg = pg_adam_cfg_default();
    cfg.lr = 0.1f;
    cfg.weight_decay = 1e-4f;
    pg_adam *opt = pg_adam_new(&cfg);
    pg_module_add_to_adam(m, opt);

    pg_lr_sched *sched = pg_lr_sched_new_cosine(0.1f, 1e-3f, 0, ITERS);

    pg_node *xn = pg_var_from_data(2, (size_t[]){N, 2}, X, false);

    float first = 0.0f, last = 0.0f;
    for (size_t it = 0; it < ITERS; it++) {
        float lr = pg_lr_sched_step(sched);
        pg_adam_set_lr(opt, lr);

        pg_node *h = pg_ag_relu(pg_ag_add(pg_ag_matmul(xn, W1), b1));
        pg_node *logits = pg_ag_add(pg_ag_matmul(h, W2), b2);
        pg_node *loss = pg_ag_cross_entropy(logits, y, N, true);

        pg_adam_zero_grad(opt);
        pg_backward(loss);

        float lv = loss->value->data[0];
        if (it == 0)
            first = lv;
        if (it == ITERS - 1)
            last = lv;

        pg_adam_step(opt);
        pg_node_free(loss); // cascades free of h, logits, ...

        if (it % 400 == 0 || it == ITERS - 1)
            printf("iter %4zu  lr %.4f  loss %.4f\n", it, lr, lv);
    }

    // final accuracy
    pg_node *h = pg_ag_relu(pg_ag_add(pg_ag_matmul(xn, W1), b1));
    pg_node *logits = pg_ag_add(pg_ag_matmul(h, W2), b2);
    size_t correct = 0;
    for (size_t i = 0; i < N; i++) {
        size_t off = i * CLS;
        float best = -1e30f;
        size_t bestc = 0;
        for (size_t c = 0; c < CLS; c++)
            if (logits->value->data[off + c] > best) {
                best = logits->value->data[off + c];
                bestc = c;
            }
        if (bestc == y[i])
            correct++;
    }
    printf("final loss %.4f  accuracy %.1f%%\n", last, 100.0f * correct / N);
    printf("loss %s (%.3f -> %.3f)\n", last < first ? "decreased" : "DID NOT DECREASE",
           first, last);

    pg_node_free(logits);
    pg_node_free(h);
    pg_node_free(xn);
    pg_module_free(m); // frees W1,b1,W2,b2
    pg_adam_free(opt);
    pg_lr_sched_free(sched);
    return 0;
}
