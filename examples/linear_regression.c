#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/autograd/autograd.h"
#include "../src/opt/sgd.h"

#define N 128
#define EPOCHS 1200
#define PRINT_EVERY 300

int main(void){
    printf("=== linear regression: y = 2.5*x + 0.7 ===\n");
    pg_seed(42);

    // data: x in [-2,2], y = 2.5*x + 0.7  (+ small noise)
    float xraw[N], yraw[N];
    for(size_t i=0;i<N;i++){
        float x = -2.0f + 4.0f * (float)i / (float)(N-1);
        // tiny noise uniform [-0.1,0.1] using simple LCG
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
        xraw[i] = x;
        yraw[i] = 2.5f * x + 0.7f + noise;
    }

    pg_node *x = pg_var_from_data(2, (size_t[]){N,1}, xraw, false);
    pg_node *y = pg_var_from_data(2, (size_t[]){N,1}, yraw, false);
    if(!x || !y) return 1;

    pg_node *w = pg_var_uniform(1, (size_t[]){1}, -1.0f, 1.0f, true);
    pg_node *b = pg_var_zeros(1, (size_t[]){1}, true);
    if(!w || !b) return 1;

    printf("init w=%.4f b=%.4f\n", pg_node_value(w)->data[0], pg_node_value(b)->data[0]);

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.05f;
    cfg.momentum = 0.0f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt, w);
    pg_sgd_add_param(opt, b);

    for(int it=0; it<EPOCHS; it++){
        // forward: y_pred = x * w + b  (broadcast scalar)
        pg_node *xw = pg_ag_mul(x, w);
        pg_node *pred = pg_ag_add(xw, b);
        pg_node_free(xw);

        pg_node *diff = pg_ag_sub(pred, y);
        pg_node_free(pred);
        pg_node *sq = pg_ag_mul(diff, diff);
        pg_node_free(diff);
        pg_node *loss = pg_ag_mean(sq, 0, false); // mean over N
        pg_node_free(sq);

        pg_backward(loss);

        if(it % PRINT_EVERY == 0 || it == EPOCHS-1){
            printf("iter %4d loss %.6f  w=%.4f b=%.4f\n",
                it, pg_node_value(loss)->data[0],
                pg_node_value(w)->data[0], pg_node_value(b)->data[0]);
        }

        pg_sgd_step(opt);
        pg_node_free(loss);
    }

    float wf = pg_node_value(w)->data[0];
    float bf = pg_node_value(b)->data[0];
    printf("\nlearned: w=%.4f (true 2.5)  b=%.4f (true 0.7)\n", wf, bf);

    // evaluate on a few points
    printf("\n predictions vs true:\n");
    float test_x[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for(int i=0;i<5;i++){
        float p = wf * test_x[i] + bf;
        float t = 2.5f * test_x[i] + 0.7f;
        printf("  x=%5.2f  pred=%6.3f  true=%6.3f  err=%+.3f\n",
            test_x[i], p, t, p - t);
    }

    bool ok = fabsf(wf - 2.5f) < 0.15f && fabsf(bf - 0.7f) < 0.15f;
    printf("\n%s\n", ok ? "PASS" : "FAIL (not converged, try more epochs)");

    pg_sgd_free(opt);
    pg_node_free(w); pg_node_free(b);
    pg_node_free(x); pg_node_free(y);
    return ok ? 0 : 1;
}
