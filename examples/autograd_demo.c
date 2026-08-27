#include <stdio.h>
#include <math.h>
#include "../src/autograd/autograd.h"
#include "../src/core/tensor.h"
#include "../src/opt/sgd.h"

static void print_grad(const char *name, pg_node *n){
    if(!n || !n->grad){ printf("  %s grad: (null)\n", name); return; }
    printf("  %s grad shape [", name);
    for(size_t i=0;i<n->value->ndim;i++) printf("%s%zu",(i?",":""), n->value->shape[i]);
    printf("] = ");
    pg_tensor_print(n->grad, stdout);
}

int main(void){
    printf("=== autograd basics ===\n");
    printf("Demos: scalar chain, broadcasting grad, softmax, layernorm, numeric grad check.\n");

    // 1. scalar chain: y = (w*x + c)^2
    printf("\n--- 1. scalar chain: w=2, x=3, c=5, y=w*x+c, loss=y^2 ---\n");
    {
        pg_node *w = pg_var_scalar(2.0f, true);
        pg_node *x = pg_var_scalar(3.0f, true);
        pg_node *c = pg_var_scalar(5.0f, false);
        pg_node *y = pg_ag_mul(w,x);               // 6
        pg_node *y2 = pg_ag_add(y,c);              // 11
        pg_node_free(y);
        pg_node *loss = pg_ag_mul(y2,y2);          // 121
        pg_node_free(y2);

        printf("forward loss = %g (expected 121)\n", pg_node_value(loss)->data[0]);
        pg_backward(loss);
        // analytic: dL/dw = 2*y2 * x = 66, dL/dx = 2*y2*w = 44
        printf("analytic: dL/dw=66, dL/dx=44\n");
        print_grad("w", w);
        print_grad("x", x);
        // c is not requires_grad, so no grad
        printf("w grad %s\n", fabsf(w->grad->data[0]-66.0f)<1e-3? "PASS":"FAIL");
        printf("x grad %s\n", fabsf(x->grad->data[0]-44.0f)<1e-3? "PASS":"FAIL");

        pg_node_free(loss);
        pg_node_free(w); pg_node_free(x); pg_node_free(c);
    }

    // 2. broadcasting grad: [2,3] + [3] -> mean
    // NOTE: pg_ag_mean_all has a scaling quirk for >1D (returns sum/last_dim, not sum/numel)
    // so we build the true mean as mean(mean(c,0),0)
    printf("\n--- 2. broadcasting grad: a[2,3] + b[3] -> mean ---\n");
    {
        float a_raw[6]={1,2,3,4,5,6};
        float b_raw[3]={10,20,30};
        pg_node *a = pg_var_from_data(2,(size_t[]){2,3}, a_raw, true);
        pg_node *b = pg_var_from_data(1,(size_t[]){3}, b_raw, true);
        pg_node *c = pg_ag_add(a,b); // [2,3]
        pg_node *m0 = pg_ag_mean(c, 0, false); // [3]  mean over 2 rows
        pg_node_free(c);
        pg_node *loss = pg_ag_mean(m0, 0, false); // scalar  mean over 3 cols = total/6
        pg_node_free(m0);
        printf("loss = mean(a+b) = %g (true mean)\n", pg_node_value(loss)->data[0]);
        pg_backward(loss);
        // c = a+b, loss = sum(c)/6, dL/dc=1/6
        // dL/da = 1/6 each element, dL/db = sum_rows(1/6)= 2*(1/6)=1/3
        printf("expected a grad all 0.1666, b grad all 0.3333\n");
        print_grad("a", a);
        print_grad("b", b);
        bool ok=true;
        for(size_t i=0;i<a->grad->numel;i++) if(fabsf(a->grad->data[i]-1.0f/6.0f)>1e-4) ok=false;
        for(size_t i=0;i<b->grad->numel;i++) if(fabsf(b->grad->data[i]-1.0f/3.0f)>1e-4) ok=false;
        printf("%s\n", ok?"PASS":"FAIL");
        pg_node_free(loss); pg_node_free(a); pg_node_free(b);
    }

    // 3. softmax grad demo
    printf("\n--- 3. softmax grad: x[1,4] -> softmax -> mean ---\n");
    {
        float xr[4]={1,2,3,1};
        pg_node *x = pg_var_from_data(2,(size_t[]){1,4}, xr, true);
        pg_node *s = pg_ag_softmax(x,1);
        printf("softmax(x) = "); pg_tensor_print(pg_node_value(s), stdout);
        pg_node *loss = pg_ag_mean_all(s);
        printf("loss (mean softmax) = %g (expected 0.25)\n", pg_node_value(loss)->data[0]);
        pg_backward(loss);
        // softmax mean: grad should sum to 0? Since mean of softmax = (1/N) sum softmax
        // dL/dx = softmax * (1/N - softmax_mean?) Not exactly. But we can just print.
        print_grad("x", x);
        // Check that grads sum approx 0 because softmax is normalized.
        float sum=0; for(size_t i=0;i<x->grad->numel;i++) sum+= x->grad->data[i];
        printf("sum grad %.6f (should be ~0 for softmax+mean): %s\n", sum, fabsf(sum)<1e-5? "PASS":"CHECK");
        pg_node_free(loss); pg_node_free(s); pg_node_free(x);
    }

    // 4. layernorm / rmsnorm forward + backward
    printf("\n--- 4. layernorm & rmsnorm ---\n");
    {
        pg_seed(0);
        pg_node *x = pg_var_uniform(2,(size_t[]){2,4}, -1,1,true);
        pg_node *w = pg_var_ones(1,(size_t[]){4}, true);
        pg_node *b = pg_var_zeros(1,(size_t[]){4}, true);
        printf("x:\n"); pg_tensor_print(pg_node_value(x), stdout);

        pg_node *ln = pg_ag_layernorm(x,w,b,1e-5f);
        printf("layernorm(x):\n"); pg_tensor_print(pg_node_value(ln), stdout);
        pg_node *m0 = pg_ag_mean(ln, 0, false);
        pg_node_free(ln);
        pg_node *loss = pg_ag_mean(m0, 0, false);
        pg_node_free(m0);
        pg_backward(loss);
        printf("after backward:\n");
        print_grad("x", x);
        print_grad("w", w);
        print_grad("b", b);
        pg_node_free(loss);
        // reset grads for rmsnorm: need new nodes or zero? We'll just create fresh.
        pg_node_free(x); pg_node_free(w); pg_node_free(b);

        x = pg_var_uniform(2,(size_t[]){2,4}, -1,1,true);
        w = pg_var_ones(1,(size_t[]){4}, true);
        printf("\nnew x for rmsnorm:\n"); pg_tensor_print(pg_node_value(x), stdout);
        pg_node *rn = pg_ag_rmsnorm(x,w,1e-5f);
        printf("rmsnorm(x):\n"); pg_tensor_print(pg_node_value(rn), stdout);
        pg_node *m1 = pg_ag_mean(rn, 0, false); pg_node_free(rn);
        pg_node *loss2 = pg_ag_mean(m1, 0, false); pg_node_free(m1);
        pg_backward(loss2);
        print_grad("x", x);
        print_grad("w", w);
        pg_node_free(loss2); pg_node_free(x); pg_node_free(w);
        // b was already freed
    }

    // 5. numeric grad check vs analytic
    printf("\n--- 5. numeric grad check (tiny linear model) ---\n");
    {
        // model: y = relu( x @ w + b ), loss = mean((y - target)^2)
        // x [2,3], w [3,2], b [2], target [2,2]
        pg_seed(42);
        float xraw[6]={0.5f,-0.3f,0.8f, -0.2f,1.0f,0.1f};
        float traw[4]={1,0, 0,1};
        pg_node *x = pg_var_from_data(2,(size_t[]){2,3}, xraw, false);
        pg_node *target = pg_var_from_data(2,(size_t[]){2,2}, traw, false);
        pg_node *w = pg_var_uniform(2,(size_t[]){3,2}, -0.5f,0.5f,true);
        pg_node *b = pg_var_zeros(1,(size_t[]){2}, true);

        // keep copies of w for numeric
        pg_tensor *w0 = pg_tensor_clone(pg_node_value(w));

        // analytic grad via autograd (true mean via two reductions)
        pg_node *mm = pg_ag_matmul(x,w);
        pg_node *z = pg_ag_add(mm,b);
        pg_node_free(mm);
        pg_node *h = pg_ag_relu(z);
        pg_node_free(z);
        pg_node *diff = pg_ag_sub(h,target);
        pg_node_free(h);
        pg_node *sq = pg_ag_mul(diff,diff);
        pg_node_free(diff);
        pg_node *m0 = pg_ag_mean(sq, 0, false); pg_node_free(sq);
        pg_node *loss = pg_ag_mean(m0, 0, false); pg_node_free(m0);
        float loss_val = pg_node_value(loss)->data[0];
        pg_backward(loss);
        printf("loss %.6f\n", loss_val);
        printf("analytic w grad:\n"); pg_tensor_print(w->grad, stdout);
        printf("analytic b grad: "); pg_tensor_print(b->grad, stdout);

        // numeric check for first 2 elements of w and b
        float eps = 1e-3f;
        for(int iter=0; iter<3; iter++){
            size_t idx = (size_t)iter; // flat index into w
            if(idx >= w->value->numel) break;
            float orig = w0->data[idx];
            // loss(w+eps)
            pg_node_value(w)->data[idx] = orig + eps;
            pg_node *mm1 = pg_ag_matmul(x,w);
            pg_node *z1 = pg_ag_add(mm1,b);
            pg_node_free(mm1);
            pg_node *h1 = pg_ag_relu(z1); pg_node_free(z1);
            pg_node *d1 = pg_ag_sub(h1,target); pg_node_free(h1);
            pg_node *sq1 = pg_ag_mul(d1,d1); pg_node_free(d1);
            pg_node *m1a = pg_ag_mean(sq1,0,false); pg_node_free(sq1);
            pg_node *l1 = pg_ag_mean(m1a,0,false); pg_node_free(m1a);
            float p1 = pg_node_value(l1)->data[0];
            pg_node_free(l1);
            // loss(w-eps)
            pg_node_value(w)->data[idx] = orig - eps;
            pg_node *mm2 = pg_ag_matmul(x,w);
            pg_node *z2 = pg_ag_add(mm2,b);
            pg_node_free(mm2);
            pg_node *h2 = pg_ag_relu(z2); pg_node_free(z2);
            pg_node *d2 = pg_ag_sub(h2,target); pg_node_free(h2);
            pg_node *sq2 = pg_ag_mul(d2,d2); pg_node_free(d2);
            pg_node *m2a = pg_ag_mean(sq2,0,false); pg_node_free(sq2);
            pg_node *l2 = pg_ag_mean(m2a,0,false); pg_node_free(m2a);
            float p2 = pg_node_value(l2)->data[0];
            pg_node_free(l2);

            // restore
            pg_node_value(w)->data[idx] = orig;

            float numeric = (p1 - p2) / (2*eps);
            float analytic = w->grad->data[idx];
            float diffv = fabsf(numeric-analytic);
            float rel = diffv / (fabsf(numeric)+fabsf(analytic)+1e-8f);
            printf(" w[%zu] numeric %.6f analytic %.6f diff %.2e %s\n",
                idx, numeric, analytic, diffv, (rel < 1e-2 || diffv < 5e-3)?"PASS":"FAIL");
        }
        // check b
        pg_tensor *b0 = pg_tensor_clone(pg_node_value(b));
        for(size_t idx=0; idx<b->value->numel && idx<2; idx++){
            float orig = b0->data[idx];
            pg_node_value(b)->data[idx]=orig+eps;
            pg_node *mm1=pg_ag_matmul(x,w); pg_node *z1=pg_ag_add(mm1,b); pg_node_free(mm1);
            pg_node *h1=pg_ag_relu(z1); pg_node_free(z1);
            pg_node *d1=pg_ag_sub(h1,target); pg_node_free(h1);
            pg_node *sq1=pg_ag_mul(d1,d1); pg_node_free(d1);
            pg_node *m1b = pg_ag_mean(sq1,0,false); pg_node_free(sq1);
            pg_node *l1=pg_ag_mean(m1b,0,false); pg_node_free(m1b);
            float p1=pg_node_value(l1)->data[0]; pg_node_free(l1);
            pg_node_value(b)->data[idx]=orig-eps;
            pg_node *mm2=pg_ag_matmul(x,w); pg_node *z2=pg_ag_add(mm2,b); pg_node_free(mm2);
            pg_node *h2=pg_ag_relu(z2); pg_node_free(z2);
            pg_node *d2=pg_ag_sub(h2,target); pg_node_free(h2);
            pg_node *sq2=pg_ag_mul(d2,d2); pg_node_free(d2);
            pg_node *m2b=pg_ag_mean(sq2,0,false); pg_node_free(sq2);
            pg_node *l2=pg_ag_mean(m2b,0,false); pg_node_free(m2b);
            float p2=pg_node_value(l2)->data[0]; pg_node_free(l2);
            pg_node_value(b)->data[idx]=orig;
            float numeric=(p1-p2)/(2*eps);
            float analytic=b->grad->data[idx];
            float diffv=fabsf(numeric-analytic);
            float rel = diffv/(fabsf(numeric)+fabsf(analytic)+1e-8f);
            printf(" b[%zu] numeric %.6f analytic %.6f diff %.2e %s\n",
                idx, numeric, analytic, diffv, (rel<1e-2||diffv<5e-3)?"PASS":"FAIL");
        }

        pg_tensor_free(w0); pg_tensor_free(b0);
        pg_node_free(loss); pg_node_free(w); pg_node_free(b);
        pg_node_free(x); pg_node_free(target);
    }

    printf("\nautograd demo finished.\n");
    return 0;
}
