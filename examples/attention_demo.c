#include <stdio.h>
#include <math.h>

#include "../src/core/tensor.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/matmul.h"
#include "../src/ops/activations.h"
#include "../src/ops/norm.h"
#include "../src/autograd/autograd.h"

static int argmax_row(const float *p, size_t n){
    int best=0; float bv=p[0];
    for(size_t i=1;i<n;i++) if(p[i]>bv){bv=p[i]; best=(int)i;}
    return best;
}

int main(void){
    printf("=== attention demo: scaled dot-product + layernorm ===\n");
    pg_seed(42);

    const size_t SEQ = 4;
    const size_t DIM = 8;

    // Input X [SEQ, DIM] random
    pg_tensor *X = pg_tensor_uniform(2,(size_t[]){SEQ,DIM}, -1.0f, 1.0f);
    printf("X [%zu,%zu]:\n", SEQ,DIM);
    pg_tensor_print(X, stdout);

    // Weights Wq,Wk,Wv [DIM,DIM] (for simplicity identity-ish scaled)
    pg_tensor *Wq = pg_tensor_uniform(2,(size_t[]){DIM,DIM}, -0.5f, 0.5f);
    pg_tensor *Wk = pg_tensor_uniform(2,(size_t[]){DIM,DIM}, -0.5f, 0.5f);
    pg_tensor *Wv = pg_tensor_uniform(2,(size_t[]){DIM,DIM}, -0.5f, 0.5f);

    // Q = X @ Wq, K = X @ Wk, V = X @ Wv  -> [SEQ,DIM]
    pg_tensor *Q = pg_matmul(X, Wq);
    pg_tensor *K = pg_matmul(X, Wk);
    pg_tensor *V = pg_matmul(X, Wv);
    printf("\nQ [SEQ,DIM]:\n"); pg_tensor_print(Q, stdout);

    // K^T [DIM,SEQ]  (matmul requires contiguous last dim, so we materialize)
    pg_tensor *Kt = pg_tensor_empty(2, (size_t[]){DIM, SEQ});
    for(size_t i=0;i<SEQ;i++)
        for(size_t j=0;j<DIM;j++)
            Kt->data[j*SEQ + i] = K->data[i*DIM + j];
    printf("K^T [%zu,%zu] (contiguous transpose):\n", Kt->shape[0], Kt->shape[1]);
    // Scores = Q @ Kt / sqrt(DIM)  -> [SEQ,SEQ]
    pg_tensor *scores = pg_matmul(Q, Kt);
    float scale = 1.0f / sqrtf((float)DIM);
    for(size_t i=0;i<scores->numel;i++) scores->data[i] *= scale;
    printf("\nscores QK^T/sqrt(d) [%zu,%zu]:\n", scores->shape[0], scores->shape[1]);
    pg_tensor_print(scores, stdout);

    // Softmax per row (attention weights)
    pg_tensor *attn = pg_softmax(scores, 1);
    printf("\nattention weights (softmax rows sum ~1):\n");
    pg_tensor_print(attn, stdout);
    for(size_t i=0;i<SEQ;i++){
        double s=0; for(size_t j=0;j<SEQ;j++) s+= attn->data[i*SEQ+j];
        printf(" row %zu sum %.5f\n", i, s);
    }

    // Out = attn @ V  -> [SEQ,DIM]
    pg_tensor *Out = pg_matmul(attn, V);
    printf("\nOut = attn @ V [%zu,%zu]:\n", Out->shape[0], Out->shape[1]);
    pg_tensor_print(Out, stdout);

    // Residual + layernorm
    pg_tensor *resid = pg_add(Out, X);
    printf("\nresidual Out+X:\n"); pg_tensor_print(resid, stdout);

    pg_tensor *ln_w = pg_tensor_ones(1,(size_t[]){DIM});
    pg_tensor *ln_b = pg_tensor_zeros(1,(size_t[]){DIM});
    pg_tensor *normed = pg_layernorm(resid, ln_w, ln_b, 1e-5f);
    printf("\nlayernorm(residual):\n"); pg_tensor_print(normed, stdout);

    // Show per-row mean/var after norm (should be ~0 / 1)
    printf("\nlayernorm stats per row:\n");
    for(size_t i=0;i<SEQ;i++){
        double sum=0; for(size_t j=0;j<DIM;j++) sum+= normed->data[i*DIM+j];
        double mean=sum/DIM;
        double var=0; for(size_t j=0;j<DIM;j++){ double d=normed->data[i*DIM+j]-mean; var+=d*d; } var/=DIM;
        printf(" row %zu mean %+.4f var %.4f\n", i, mean, var);
    }

    // --- autograd version of same attention (learnable) ---
    printf("\n--- autograd: tiny training step through attention ---\n");
    {
        // make X a parameter that we can optimize to match a target
        pg_node *xn = pg_var_from_tensor(X, true);
        pg_node *wq = pg_var_uniform(2,(size_t[]){DIM,DIM}, -0.5f,0.5f,true);
        pg_node *wk = pg_var_uniform(2,(size_t[]){DIM,DIM}, -0.5f,0.5f,true);
        pg_node *wv = pg_var_uniform(2,(size_t[]){DIM,DIM}, -0.5f,0.5f,true);
        // target = layernorm(X) (just to have something)
        pg_tensor *targ_t = pg_layernorm(X, ln_w, ln_b, 1e-5f);
        pg_node *target = pg_var_from_tensor(targ_t, false);

        // forward with autograd ops (using matmul + layernorm + etc manually via autograd helpers)
        // we reuse plain tensor logic but wrap via autograd nodes for the linear projections.
        // For K^T we need transpose trick: since autograd matmul does not have transpose flag,
        // we do it manually via permute not tracked. So we demonstrate a simplified attention
        // where we do QK^T via eager transpose inside autograd? For demo we just show that
        //layernorm and matmul have grads.

        // simpler: y = layernorm( relu( X @ Wq ) )
        pg_node *q = pg_ag_matmul(xn, wq);
        pg_node *qr = pg_ag_relu(q); pg_node_free(q);
        // layernorm requires weight/bias nodes
        pg_node *lnw = pg_var_ones(1,(size_t[]){DIM}, true);
        pg_node *lnb = pg_var_zeros(1,(size_t[]){DIM}, true);
        pg_node *y = pg_ag_layernorm(qr, lnw, lnb, 1e-5f); pg_node_free(qr);
        // loss = mean((y - target)^2)  correct mean via two steps for [SEQ,DIM]
        pg_node *diff = pg_ag_sub(y, target); pg_node_free(y);
        pg_node *sq = pg_ag_mul(diff,diff); pg_node_free(diff);
        pg_node *m0 = pg_ag_mean(sq, 0, false); pg_node_free(sq);
        pg_node *loss = pg_ag_mean(m0, 0, false); pg_node_free(m0);
        printf("loss before %.6f\n", pg_node_value(loss)->data[0]);
        pg_backward(loss);
        printf("grad wq norm: ");
        double gsum=0; for(size_t i=0;i<wq->grad->numel;i++) gsum+= fabs(wq->grad->data[i]);
        printf("%.4f mean abs\n", gsum / wq->grad->numel);
        printf("grad xn [first row]: ");
        for(size_t j=0;j<4;j++) printf("%+.3f ", xn->grad->data[j]);
        printf("\n");

        pg_node_free(loss);
        pg_node_free(xn); pg_node_free(wq); pg_node_free(wk); pg_node_free(wv);
        pg_node_free(target); pg_node_free(lnw); pg_node_free(lnb);
        pg_tensor_free(targ_t);
    }

    // cleanup
    pg_tensor_free(X); pg_tensor_free(Wq); pg_tensor_free(Wk); pg_tensor_free(Wv);
    pg_tensor_free(Q); pg_tensor_free(K); pg_tensor_free(V);
    pg_tensor_free(Kt); // view free
    pg_tensor_free(scores); pg_tensor_free(attn); pg_tensor_free(Out);
    pg_tensor_free(resid); pg_tensor_free(ln_w); pg_tensor_free(ln_b); pg_tensor_free(normed);

    printf("\nattention demo finished.\n");
    return 0;
}
