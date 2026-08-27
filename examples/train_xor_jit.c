#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <time.h>
#include "../src/autograd/autograd.h"
#include "../src/core/tensor.h"
#include "../src/jit/jit.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/matmul.h"
#include "../src/ops/activations.h"
#include "../src/opt/sgd.h"

/*
 * XOR training with JIT-accelerated elementwise part.
 * Demonstrates how (add + tanh) is fused into a single kernel.
 * MatMul stays via pg_gemm (AVX2-optimized), bias+activation is JIT.
 */

#define EPOCHS 3000
#define PRINT_EVERY 1000

/* JIT for bias + tanh: out = tanh(matmul + bias) */
typedef struct {
    pg_jit_exe *add_tanh; // [N,8] + [8] -> tanh
    pg_jit_exe *add;      // [N,1] + [1] -> out
} jit_cache_t;

static jit_cache_t g_jit = {0};

static void jit_init(void){
    // add_tanh: [4,8] + [8] -> tanh -> [4,8]  (for w1)
    pg_jit_graph *g1 = pg_jit_graph_new();
    int a = pg_jit_add_input(g1, 2, (size_t[]){4,8});
    int b = pg_jit_add_input(g1, 1, (size_t[]){8});
    int t = pg_jit_add(g1,a,b);
    int h = pg_jit_tanh(g1,t);
    pg_jit_mark_output(g1,h);
    g_jit.add_tanh = pg_jit_compile(g1);
    pg_jit_graph_free(g1);
    if(!g_jit.add_tanh) printf("JIT add+tanh compile failed: %s\n", pg_jit_last_error());

    // add: [4,1] + [1] -> [4,1] (for w2 bias)
    pg_jit_graph *g2 = pg_jit_graph_new();
    int c = pg_jit_add_input(g2, 2, (size_t[]){4,1});
    int d = pg_jit_add_input(g2, 1, (size_t[]){1});
    int y = pg_jit_add(g2,c,d);
    pg_jit_mark_output(g2,y);
    g_jit.add = pg_jit_compile(g2);
    pg_jit_graph_free(g2);
    if(!g_jit.add) printf("JIT add compile failed: %s\n", pg_jit_last_error());
}

static void jit_free(void){
    pg_jit_exe_free(g_jit.add_tanh);
    pg_jit_exe_free(g_jit.add);
    pg_jit_cache_clear();
}

/* JIT-accelerated forward: matmul stays eager, add+act is JIT */
static pg_node *forward_jit(pg_node *x, pg_node *w1, pg_node *b1, pg_node *w2, pg_node *b2){
    pg_node *mm1 = pg_ag_matmul(x, w1); // [4,8]
    // JIT: mm1 + b1 -> tanh
    pg_tensor *mm1_t = pg_node_value(mm1);
    pg_tensor *b1_t = pg_node_value(b1);
    const pg_tensor *ins1[2]={mm1_t, b1_t};
    pg_tensor *h_t = pg_jit_run_single(g_jit.add_tanh, ins1, 2);
    // wrap into pg_node for autograd (no JIT autograd, do it manually via eager ops for gradients)
    // Simple path: use eager pg_add+pg_tanh for the graph, but for speed demo compare inference.
    // Here for training we keep the eager graph so backward is correct.
    // Therefore for training forward_jit is not really used; the demo shows inference speedup.
    pg_tensor_free(h_t);
    pg_node_free(mm1);
    // fallback to eager for correct grad
    pg_node *mm1e = pg_ag_matmul(x,w1);
    pg_node *z1 = pg_ag_add(mm1e,b1);
    pg_node_free(mm1e);
    pg_node *h = pg_ag_tanh(z1);
    pg_node_free(z1);
    pg_node *mm2 = pg_ag_matmul(h,w2);
    pg_node_free(h);
    pg_node *y = pg_ag_add(mm2,b2);
    pg_node_free(mm2);
    return y;
}

/* inference with JIT (no autograd, plain tensors) */
static pg_tensor *infer_jit(const pg_tensor *x, const pg_tensor *w1, const pg_tensor *b1,
                            const pg_tensor *w2, const pg_tensor *b2){
    // mm1 = x @ w1  [4,2]@[2,8]=[4,8]
    extern pg_tensor *pg_matmul(const pg_tensor*,const pg_tensor*);
    pg_tensor *mm1 = pg_matmul(x,w1);
    const pg_tensor *ins1[2]={mm1,b1};
    pg_tensor *h = pg_jit_run_single(g_jit.add_tanh, ins1, 2);
    pg_tensor_free(mm1);
    pg_tensor *mm2 = pg_matmul(h,w2); // [4,8]@[8,1]=[4,1]
    pg_tensor_free(h);
    const pg_tensor *ins2[2]={mm2,b2};
    pg_tensor *y = pg_jit_run_single(g_jit.add, ins2, 2);
    pg_tensor_free(mm2);
    return y;
}

static pg_node *mse_loss(pg_node *y, pg_node *target){
    pg_node *d=pg_ag_sub(y,target);
    pg_node *sq=pg_ag_mul(d,d);
    pg_node_free(d);
    pg_node *loss=pg_ag_mean_all(sq);
    pg_node_free(sq);
    return loss;
}

int main(void){
    pg_seed(7);
    jit_init();
    if(!g_jit.add_tanh || !g_jit.add) return 1;

    float xraw[8]={0,0,0,1,1,0,1,1};
    float yraw[4]={0,1,1,0};
    pg_node *x=pg_var_from_data(2,(size_t[]){4,2},xraw,false);
    pg_node *y=pg_var_from_data(2,(size_t[]){4,1},yraw,false);
    pg_node *w1=pg_var_uniform(2,(size_t[]){2,8},-1,1,true);
    pg_node *b1=pg_var_zeros(1,(size_t[]){8},true);
    pg_node *w2=pg_var_uniform(2,(size_t[]){8,1},-1,1,true);
    pg_node *b2=pg_var_zeros(1,(size_t[]){1},true);

    pg_sgd_cfg cfg=pg_sgd_cfg_default(); cfg.lr=0.1f; cfg.momentum=0.9f;
    pg_sgd *opt=pg_sgd_new(&cfg);
    pg_sgd_add_param(opt,w1); pg_sgd_add_param(opt,b1); pg_sgd_add_param(opt,w2); pg_sgd_add_param(opt,b2);

    // training with plain eager (so gradients are correct)
    for(int it=0;it<EPOCHS;it++){
        pg_node *out=forward_jit(x,w1,b1,w2,b2);
        pg_node *loss=mse_loss(out,y);
        pg_node_free(out);
        pg_backward(loss);
        if(it%PRINT_EVERY==0||it==EPOCHS-1) printf("iter %5d loss %.6f (cache %zu)\n", it, pg_node_value(loss)->data[0], pg_jit_cache_size());
        pg_sgd_step(opt);
        pg_node_free(loss);
    }

    // inference benchmark: eager vs jit
    printf("\n--- inference benchmark (10000 runs) ---\n");
    pg_tensor *xt=pg_tensor_from_data(2,(size_t[]){4,2},xraw);
    pg_tensor *w1t=pg_node_value(w1);
    pg_tensor *b1t=pg_node_value(b1);
    pg_tensor *w2t=pg_node_value(w2);
    pg_tensor *b2t=pg_node_value(b2);
    // eager infer
    struct timespec ts0,ts1;
    clock_gettime(CLOCK_MONOTONIC,&ts0);
    for(int i=0;i<10000;i++){
        pg_tensor *mm1=pg_matmul(xt,w1t);
        pg_tensor *z1=pg_add(mm1,b1t); pg_tensor_free(mm1);
        pg_tensor *h=pg_tanh(z1); pg_tensor_free(z1);
        pg_tensor *mm2=pg_matmul(h,w2t); pg_tensor_free(h);
        pg_tensor *out=pg_add(mm2,b2t); pg_tensor_free(mm2); pg_tensor_free(out);
    }
    clock_gettime(CLOCK_MONOTONIC,&ts1);
    double eager_ms=(ts1.tv_sec-ts0.tv_sec)*1e3 + (ts1.tv_nsec-ts0.tv_nsec)/1e6;
    clock_gettime(CLOCK_MONOTONIC,&ts0);
    for(int i=0;i<10000;i++){
        pg_tensor *out=infer_jit(xt,w1t,b1t,w2t,b2t);
        pg_tensor_free(out);
    }
    clock_gettime(CLOCK_MONOTONIC,&ts1);
    double jit_ms=(ts1.tv_sec-ts0.tv_sec)*1e3 + (ts1.tv_nsec-ts0.tv_nsec)/1e6;
    printf("eager %.1f ms  jit %.1f ms  speedup %.2fx\n", eager_ms, jit_ms, eager_ms/jit_ms);

    // final predictions via jit
    printf("\n predictions (jit):\n");
    pg_tensor *out=infer_jit(xt,w1t,b1t,w2t,b2t);
    int correct=0; const float targets[4]={0,1,1,0};
    for(size_t i=0;i<4;i++){
        float p=out->data[i];
        int cls=p>0.5f;
        correct+=cls==(int)targets[i];
        printf("  (%.0f,%.0f) -> %.4f\n", xraw[i*2], xraw[i*2+1], p);
    }
    printf("accuracy %d/4\n", correct);
    pg_tensor_free(out); pg_tensor_free(xt);

    pg_sgd_free(opt);
    pg_node_free(w1); pg_node_free(b1); pg_node_free(w2); pg_node_free(b2); pg_node_free(x); pg_node_free(y);
    jit_free();
    return correct!=4;
}
