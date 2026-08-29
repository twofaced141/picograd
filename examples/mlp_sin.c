#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../src/autograd/autograd.h"
#include "../src/opt/sgd.h"

#define N 128
#define HIDDEN 32
#define EPOCHS 4000
#define PRINT_EVERY 800

static pg_node *forward(pg_node *x, pg_node *w1, pg_node *b1, pg_node *w2, pg_node *b2){
    pg_node *mm1 = pg_ag_matmul(x, w1);          // [N,32]
    pg_node *z1  = pg_ag_add(mm1, b1);           // broadcast [32]
    pg_node_free(mm1);
    pg_node *h   = pg_ag_tanh(z1);
    pg_node_free(z1);
    pg_node *mm2 = pg_ag_matmul(h, w2);          // [N,1]
    pg_node_free(h);
    pg_node *out = pg_ag_add(mm2, b2);
    pg_node_free(mm2);
    return out;
}

int main(void){
    printf("=== MLP sin approximation: 1 -> 32 tanh -> 1 ===\n");
    pg_seed(0);

    float xraw[N], yraw[N];
    for(size_t i=0;i<N;i++){
        float x = - (float)M_PI + 2.0f*(float)M_PI * (float)i / (float)(N-1);
        xraw[i] = x;
        yraw[i] = sinf(x);
    }
    pg_node *x = pg_var_from_data(2, (size_t[]){N,1}, xraw, false);
    pg_node *y = pg_var_from_data(2, (size_t[]){N,1}, yraw, false);

    pg_node *w1 = pg_var_uniform(2, (size_t[]){1, HIDDEN}, -1.0f, 1.0f, true);
    pg_node *b1 = pg_var_zeros(1, (size_t[]){HIDDEN}, true);
    pg_node *w2 = pg_var_uniform(2, (size_t[]){HIDDEN, 1}, -1.0f, 1.0f, true);
    pg_node *b2 = pg_var_zeros(1, (size_t[]){1}, true);
    if(!x||!y||!w1||!b1||!w2||!b2) return 1;

    // Xavier-like scale for tanh
    // divide a bit for stability
    for(size_t i=0;i<w1->value->numel;i++) w1->value->data[i] *= 0.5f;
    for(size_t i=0;i<w2->value->numel;i++) w2->value->data[i] *= 0.5f;

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.03f;
    cfg.momentum = 0.9f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt, w1); pg_sgd_add_param(opt, b1);
    pg_sgd_add_param(opt, w2); pg_sgd_add_param(opt, b2);

    for(int it=0; it<EPOCHS; it++){
        pg_node *pred = forward(x,w1,b1,w2,b2);
        pg_node *diff = pg_ag_sub(pred, y);
        pg_node_free(pred);
        pg_node *sq = pg_ag_mul(diff,diff);
        pg_node_free(diff);
        // correct overall mean: mean over N (axis 0). For [N,1] this is the true mean
        pg_node *loss = pg_ag_mean(sq, 0, false); // [1]
        pg_node_free(sq);

        pg_backward(loss);
        if(it%PRINT_EVERY==0 || it==EPOCHS-1)
            printf("iter %4d loss %.6f\n", it, pg_node_value(loss)->data[0]);
        pg_sgd_step(opt);
        pg_node_free(loss);
    }

    // evaluation
    printf("\n  x        pred     true    err\n");
    float max_err = 0;
    pg_node *pred = forward(x,w1,b1,w2,b2);
    float *pd = pg_node_value(pred)->data;
    for(size_t i=0;i<N;i+= N/8){
        float e = fabsf(pd[i] - yraw[i]);
        if(e>max_err) max_err=e;
        printf(" %6.3f  %7.4f  %7.4f  %+.4f\n", xraw[i], pd[i], yraw[i], pd[i]-yraw[i]);
    }
    // also test 5 canonical points
    printf("\n canonical:\n");
    float pts[5] = {0.0f, (float)M_PI/2, (float)-M_PI/2, (float)M_PI, (float)-M_PI};
    pg_node *xp = pg_var_from_data(2,(size_t[]){5,1}, pts, false);
    pg_node *pr = forward(xp,w1,b1,w2,b2);
    for(int i=0;i<5;i++){
        float p = pg_node_value(pr)->data[i];
        float t = sinf(pts[i]);
        printf("  sin(%6.3f)=%6.3f pred %6.3f err %+.3f\n", pts[i], t, p, p-t);
        float e = fabsf(p-t);
        if(e>max_err) max_err=e;
    }

    printf("\nmax abs err (sampled) %.4f\n", max_err);
    bool ok = max_err < 0.18f;
    printf("%s\n", ok ? "PASS" : "FAIL");

    pg_node_free(pr); pg_node_free(xp);
    pg_node_free(pred);
    pg_sgd_free(opt);
    pg_node_free(w1); pg_node_free(b1); pg_node_free(w2); pg_node_free(b2);
    pg_node_free(x); pg_node_free(y);
    return ok?0:1;
}
