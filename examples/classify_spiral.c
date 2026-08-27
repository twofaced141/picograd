#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../src/autograd/autograd.h"
#include "../src/core/tensor.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/matmul.h"
#include "../src/ops/activations.h"
#include "../src/opt/sgd.h"

#define N_PER_CLASS 50
#define N_CLASSES 3
#define N (N_PER_CLASS * N_CLASSES)
#define HIDDEN 32
#define EPOCHS 2500
#define PRINT_EVERY 500

static pg_node *forward(pg_node *x, pg_node *w1, pg_node *b1, pg_node *w2, pg_node *b2){
    pg_node *mm1 = pg_ag_matmul(x,w1);       // [N, H]
    pg_node *z1  = pg_ag_add(mm1,b1);
    pg_node_free(mm1);
    pg_node *h   = pg_ag_relu(z1);
    pg_node_free(z1);
    pg_node *mm2 = pg_ag_matmul(h,w2);       // [N, C]
    pg_node_free(h);
    pg_node *logits = pg_ag_add(mm2,b2);
    pg_node_free(mm2);
    return logits; // [N, C]
}

static int argmax_row(const float *p, size_t n){
    int best=0; float bv=p[0];
    for(size_t i=1;i<n;i++) if(p[i]>bv){bv=p[i]; best=(int)i;}
    return best;
}

int main(void){
    printf("=== 3-blobs classification: 2 -> %d relu -> %d (softmax + cross-entropy) ===\n", HIDDEN, N_CLASSES);
    pg_seed(123);
    srand(123);

    // generate 3 gaussian blobs
    float centres[3][2] = {{ 2.0f, 2.0f},{-2.0f, 1.5f},{0.0f,-2.5f}};
    float Xraw[N*2];
    float Yraw[N*N_CLASSES]; // one-hot
    int labels[N];
    for(int c=0;c<N_CLASSES;c++){
        for(int i=0;i<N_PER_CLASS;i++){
            // Box-Muller
            float u1 = (float)rand() / (float)RAND_MAX; if(u1<1e-6f) u1=1e-6f;
            float u2 = (float)rand() / (float)RAND_MAX;
            float r = sqrtf(-2.0f*logf(u1));
            float z0 = r * cosf(2.0f*3.14159265f*u2);
            float z1 = r * sinf(2.0f*3.14159265f*u2);
            float sigma = 0.6f;
            size_t idx = (size_t)(c*N_PER_CLASS + i);
            Xraw[idx*2+0] = centres[c][0] + sigma*z0;
            Xraw[idx*2+1] = centres[c][1] + sigma*z1;
            labels[idx]=c;
            for(int k=0;k<N_CLASSES;k++) Yraw[idx*N_CLASSES + k] = (k==c)?1.0f:0.0f;
        }
    }

    // shuffle with Fisher-Yates using same seed
    for(int i=N-1;i>0;i--){
        int j = rand() % (i+1);
        // swap X
        for(int d=0;d<2;d++){ float t=Xraw[i*2+d]; Xraw[i*2+d]=Xraw[j*2+d]; Xraw[j*2+d]=t; }
        for(int k=0;k<N_CLASSES;k++){ float t=Yraw[i*N_CLASSES+k]; Yraw[i*N_CLASSES+k]=Yraw[j*N_CLASSES+k]; Yraw[j*N_CLASSES+k]=t; }
        int tl=labels[i]; labels[i]=labels[j]; labels[j]=tl;
    }

    pg_node *x = pg_var_from_data(2,(size_t[]){N,2}, Xraw, false);
    pg_node *y = pg_var_from_data(2,(size_t[]){N,N_CLASSES}, Yraw, false);

    pg_node *w1 = pg_var_uniform(2,(size_t[]){2,HIDDEN}, -1.0f,1.0f,true);
    pg_node *b1 = pg_var_zeros(1,(size_t[]){HIDDEN}, true);
    pg_node *w2 = pg_var_uniform(2,(size_t[]){HIDDEN,N_CLASSES}, -1.0f,1.0f,true);
    pg_node *b2 = pg_var_zeros(1,(size_t[]){N_CLASSES}, true);
    // scale init
    for(size_t i=0;i<w1->value->numel;i++) w1->value->data[i]*=0.5f;
    for(size_t i=0;i<w2->value->numel;i++) w2->value->data[i]*=0.5f;

    pg_sgd_cfg cfg = pg_sgd_cfg_default();
    cfg.lr = 0.08f;
    cfg.momentum = 0.9f;
    pg_sgd *opt = pg_sgd_new(&cfg);
    pg_sgd_add_param(opt,w1); pg_sgd_add_param(opt,b1);
    pg_sgd_add_param(opt,w2); pg_sgd_add_param(opt,b2);

    for(int it=0; it<EPOCHS; it++){
        pg_node *logits = forward(x,w1,b1,w2,b2);
        pg_node *sm = pg_ag_softmax(logits,1);
        pg_node_free(logits);

        // cross-entropy: -mean( sum( y * log(sm) ) )
        pg_node *lsm = pg_ag_log(sm);
        pg_node_free(sm);
        pg_node *prod = pg_ag_mul(lsm, y);
        pg_node_free(lsm);
        pg_node *sum = pg_ag_sum(prod,1,false); // [N]
        pg_node_free(prod);
        pg_node *mean = pg_ag_mean_all(sum); // scalar mean over N
        pg_node_free(sum);
        pg_node *loss = pg_ag_neg(mean);
        pg_node_free(mean);

        pg_backward(loss);

        if(it%PRINT_EVERY==0 || it==EPOCHS-1){
            // quick accuracy
            pg_node *lg = forward(x,w1,b1,w2,b2);
            pg_node *s = pg_ag_softmax(lg,1);
            pg_node_free(lg);
            float *pd = pg_node_value(s)->data;
            int correct=0;
            for(int i=0;i<N;i++) if(argmax_row(pd+i*N_CLASSES, N_CLASSES)==labels[i]) correct++;
            printf("iter %4d loss %.4f  acc %d/%d (%.1f%%)\n", it,
                pg_node_value(loss)->data[0], correct,N, 100.0f*correct/N);
            pg_node_free(s);
        }

        pg_sgd_step(opt);
        pg_node_free(loss);
    }

    // final evaluation
    pg_node *logits = forward(x,w1,b1,w2,b2);
    pg_node *sm = pg_ag_softmax(logits,1);
    pg_node_free(logits);
    float *pd = pg_node_value(sm)->data;
    int correct=0;
    printf("\n samples (first 10):\n");
    for(int i=0;i<10;i++){
        int pred = argmax_row(pd+i*N_CLASSES, N_CLASSES);
        int gt = labels[i];
        printf("  x=(%5.2f,%5.2f) pred %d true %d %s\n",
            Xraw[i*2], Xraw[i*2+1], pred, gt, pred==gt?"✓":"✗");
    }
    for(int i=0;i<N;i++) if(argmax_row(pd+i*N_CLASSES,N_CLASSES)==labels[i]) correct++;
    printf("\nfinal accuracy %d/%d  (%.1f%%)\n", correct,N, 100.0f*correct/N);
    bool ok = correct >= (int)(N*0.92f);
    printf("%s\n", ok ? "PASS" : "FAIL");

    // also demo plain tensor inference (no autograd) from learned weights
    {
        pg_tensor *xt = pg_tensor_from_data(2,(size_t[]){N,2}, Xraw);
        pg_tensor *w1t = pg_node_value(w1);
        pg_tensor *b1t = pg_node_value(b1);
        pg_tensor *w2t = pg_node_value(w2);
        pg_tensor *b2t = pg_node_value(b2);
        pg_tensor *mm1 = pg_matmul(xt,w1t);
        pg_tensor *z1 = pg_add(mm1,b1t); pg_tensor_free(mm1);
        pg_tensor *h = pg_relu(z1); pg_tensor_free(z1);
        pg_tensor *mm2 = pg_matmul(h,w2t); pg_tensor_free(h);
        pg_tensor *lg2 = pg_add(mm2,b2t); pg_tensor_free(mm2);
        pg_tensor *prob = pg_softmax(lg2,1); pg_tensor_free(lg2);
        int c2=0; for(int i=0;i<N;i++) if(argmax_row(prob->data+i*N_CLASSES,N_CLASSES)==labels[i]) c2++;
        printf("plain tensor inference accuracy %d/%d\n",c2,N);
        pg_tensor_free(xt); pg_tensor_free(prob);
    }

    pg_node_free(sm);
    pg_sgd_free(opt);
    pg_node_free(w1); pg_node_free(b1); pg_node_free(w2); pg_node_free(b2);
    pg_node_free(x); pg_node_free(y);
    return ok?0:1;
}
