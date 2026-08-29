#include <stdio.h>
#include <math.h>
#include "../src/core/tensor.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/matmul.h"
#include "../src/ops/activations.h"
#include "../src/ops/reduce.h"
#include "../src/ops/norm.h"
#include "../src/ops/index.h"
#include "../src/ops/scan.h"
#include "../src/jit/jit.h"

static void print_hdr(const char *s){
    printf("\n=== %s ===\n", s);
}

int main(void){
    pg_seed(0);
    printf("=== picograd ops showcase ===\n");

    // 1. creation
    print_hdr("1. tensor creation");
    {
        pg_tensor *eye = pg_tensor_eye(3);
        printf("eye 3x3:\n"); pg_tensor_print(eye, stdout);
        pg_tensor_free(eye);

        pg_tensor *ls = pg_tensor_linspace(0,1,5);
        printf("linspace 0..1 (5): "); pg_tensor_print(ls,stdout);
        pg_tensor_free(ls);

        pg_tensor *ar = pg_tensor_arange(0,6,1);
        printf("arange 0..6: "); pg_tensor_print(ar,stdout);
        pg_tensor_free(ar);

        pg_tensor *uni = pg_tensor_uniform(2,(size_t[]){2,3}, -1,1);
        printf("uniform [2,3]:\n"); pg_tensor_print(uni,stdout);
        pg_tensor_free(uni);

        pg_tensor *norm = pg_tensor_normal(2,(size_t[]){2,4}, 0,1);
        printf("normal [2,4] mean~0 std~1:\n"); pg_tensor_print(norm,stdout);
        pg_tensor_free(norm);
    }

    // 2. views (reshape, permute, view)
    print_hdr("2. views (no-copy)");
    {
        float raw[12]; for(int i=0;i<12;i++) raw[i]=(float)i;
        pg_tensor *a = pg_tensor_from_data(2,(size_t[]){3,4}, raw);
        printf("a [3,4]:\n"); pg_tensor_print(a,stdout);

        pg_tensor *v = pg_tensor_reshape_view(a,2,(size_t[]){2,6});
        printf("reshape_view [2,6]:\n"); pg_tensor_print(v,stdout);

        pg_tensor *p = pg_tensor_permute_view(a, (size_t[]){1,0});
        printf("permute [1,0] -> [4,3] (view, strides [%zu,%zu]):\n", p->stride[0], p->stride[1]);
        // pg_tensor_print assumes contiguous last dim, so for non-contiguous permute we show via pg_tensor_get
        printf("p logical values (via get):\n");
        for(size_t i=0;i<p->shape[0];i++){
            printf(" [");
            for(size_t j=0;j<p->shape[1];j++){
                size_t idx[2]={i,j};
                printf("%s%g", j?", ":"", pg_tensor_get(p, idx));
            }
            printf("]%s", i+1<p->shape[0]?"\n":"\n");
        }
        // cumsum demo on contiguous copy (permute views are non-contiguous, clone is raw-buffer copy)
        pg_tensor *ac = pg_tensor_clone(a);
        pg_tensor *cs = pg_cumsum(ac,0);
        printf("cumsum(a, axis=0) [%zu,%zu]:\n", cs->shape[0], cs->shape[1]); pg_tensor_print(cs,stdout);

        printf("reshape inplace a to [4,3] : %s\n", pg_tensor_reshape(a,2,(size_t[]){4,3})?"ok":"fail");
        pg_tensor_print(a,stdout);

        pg_tensor_free(cs); pg_tensor_free(ac); pg_tensor_free(p); pg_tensor_free(v); pg_tensor_free(a);
    }

    // 3. elementwise + broadcasting
    print_hdr("3. elementwise + broadcasting");
    {
        pg_tensor *a = pg_tensor_from_data(2,(size_t[]){2,3}, (float[]){1,2,3,4,5,6});
        pg_tensor *b = pg_tensor_from_data(1,(size_t[]){3}, (float[]){10,20,30});
        printf("a [2,3]:\n"); pg_tensor_print(a,stdout);
        printf("b [3]: "); pg_tensor_print(b,stdout);

        pg_tensor *s = pg_add(a,b);
        printf("a + b (broadcast [2,3]+[3]):\n"); pg_tensor_print(s,stdout);

        pg_tensor *col = pg_tensor_from_data(2,(size_t[]){2,1}, (float[]){100,200});
        pg_tensor *pr = pg_mul(col, pg_tensor_from_data(2,(size_t[]){1,3}, (float[]){1,2,3}));
        printf("col [2,1] * row [1,3] -> [2,3]:\n"); pg_tensor_print(pr,stdout);

        pg_tensor *e = pg_exp(pg_tensor_zeros(1,(size_t[]){1}));
        printf("exp(0)=%g\n", e->data[0]);

        float pi6=3.14159265f/6.0f;
        pg_tensor *ang = pg_tensor_full(1,(size_t[]){1}, pi6);
        pg_tensor *sn = pg_sin(ang), *cs = pg_cos(ang);
        printf("sin(pi/6)=%g cos=%g\n", sn->data[0], cs->data[0]);

        pg_tensor *sq = pg_sqrt(pg_tensor_full(1,(size_t[]){4}, 4.0f));
        printf("sqrt([4,4,4,4])="); pg_tensor_print(sq,stdout);

        pg_tensor *cl = pg_clamp(a, 2.0f, 4.5f);
        printf("clamp(a,2,4.5):\n"); pg_tensor_print(cl,stdout);

        pg_tensor_free(a); pg_tensor_free(b); pg_tensor_free(s);
        pg_tensor_free(col); pg_tensor_free(pr); pg_tensor_free(e);
        pg_tensor_free(ang); pg_tensor_free(sn); pg_tensor_free(cs);
        pg_tensor_free(sq); pg_tensor_free(cl);
    }

    // 4. matmul family
    print_hdr("4. matmul / bmm / addmm / tensordot");
    {
        pg_tensor *A = pg_tensor_from_data(2,(size_t[]){2,3}, (float[]){1,2,3,4,5,6});
        pg_tensor *B = pg_tensor_from_data(2,(size_t[]){3,2}, (float[]){7,8,9,10,11,12});
        pg_tensor *C = pg_matmul(A,B);
        printf("A [2,3] @ B [3,2] = C [2,2]:\n"); pg_tensor_print(C,stdout);
        // expected: [[58,64],[139,154]]
        pg_tensor_free(C);

        // bmm
        pg_seed(1);
        pg_tensor *ba = pg_tensor_uniform(3,(size_t[]){2,2,3}, -1,1);
        pg_tensor *bb = pg_tensor_uniform(3,(size_t[]){2,3,4}, -1,1);
        pg_tensor *bc = pg_bmm(ba,bb);
        printf("bmm [2,2,3] @ [2,3,4] -> [%zu,%zu,%zu]:\n", bc->shape[0], bc->shape[1], bc->shape[2]);
        pg_tensor_print(bc,stdout);

        // addmm: D = 0.5*inp + 2*A*B
        pg_tensor *inp = pg_tensor_full(2,(size_t[]){2,2}, 1.0f);
        pg_tensor *am = pg_addmm(inp,A,B,2.0f,0.5f);
        printf("addmm inp*0.5 + (A@B)*2:\n"); pg_tensor_print(am,stdout);

        // tensordot: contract 1 dim
        pg_tensor *da = pg_tensor_uniform(2,(size_t[]){3,4}, -1,1);
        pg_tensor *db = pg_tensor_uniform(2,(size_t[]){4,5}, -1,1);
        size_t axa[1]={1}, axb[1]={0};
        pg_tensor *td = pg_tensordot(da,db,1,axa,axb);
        printf("tensordot [3,4]·[4,5] (axes 1,0) -> [%zu,%zu]:\n", td->shape[0], td->shape[1]);
        pg_tensor_print(td,stdout);

        pg_tensor_free(A); pg_tensor_free(B);
        pg_tensor_free(ba); pg_tensor_free(bb); pg_tensor_free(bc);
        pg_tensor_free(inp); pg_tensor_free(am);
        pg_tensor_free(da); pg_tensor_free(db); pg_tensor_free(td);
    }

    // 5. activations
    print_hdr("5. activations & softmax");
    {
        pg_tensor *x = pg_tensor_from_data(2,(size_t[]){2,4}, (float[]){-2,-0.5,0.5,2, 1,-1,0,0.3f});
        pg_tensor *r = pg_relu(x);
        printf("relu:\n"); pg_tensor_print(r,stdout);
        pg_tensor *s = pg_sigmoid(x);
        printf("sigmoid:\n"); pg_tensor_print(s,stdout);
        pg_tensor *t = pg_tanh(x);
        printf("tanh:\n"); pg_tensor_print(t,stdout);
        pg_tensor *g = pg_gelu(x);
        printf("gelu:\n"); pg_tensor_print(g,stdout);
        pg_tensor *sm = pg_softmax(x,1);
        printf("softmax axis=1 (row sums ~1):\n"); pg_tensor_print(sm,stdout);
        // verify row sums
        for(size_t i=0;i<2;i++){
            double sum=0; for(size_t j=0;j<4;j++) sum+= sm->data[i*4+j];
            printf("  row %zu sum %.6f\n", i, sum);
        }
        pg_tensor_free(x); pg_tensor_free(r); pg_tensor_free(s); pg_tensor_free(t);
        pg_tensor_free(g); pg_tensor_free(sm);
    }

    // 6. reductions
    print_hdr("6. reductions (sum/mean/max/min/var/std/argmax)");
    {
        pg_tensor *a = pg_tensor_uniform(2,(size_t[]){3,4}, -2,2);
        printf("a [3,4]:\n"); pg_tensor_print(a,stdout);
        pg_tensor *su0 = pg_sum(a,0,false); // [4]
        printf("sum axis=0 -> [4]: "); pg_tensor_print(su0,stdout);
        pg_tensor *mn1 = pg_mean(a,1,true); // [3,1]
        printf("mean axis=1 keepdim:\n"); pg_tensor_print(mn1,stdout);
        pg_tensor *mx = pg_max(a,1,false);
        printf("max axis=1 -> [3]: "); pg_tensor_print(mx,stdout);
        pg_tensor *vr = pg_var(a,0,false,0);
        printf("var axis=0 ddof0: "); pg_tensor_print(vr,stdout);
        pg_tensor *am = pg_argmax(a,1,false);
        printf("argmax axis=1: "); pg_tensor_print(am,stdout);
        pg_tensor_free(a); pg_tensor_free(su0); pg_tensor_free(mn1);
        pg_tensor_free(mx); pg_tensor_free(vr); pg_tensor_free(am);
    }

    // 7. norm
    print_hdr("7. layernorm / rmsnorm");
    {
        pg_tensor *x = pg_tensor_uniform(2,(size_t[]){2,4}, -1,1);
        pg_tensor *w = pg_tensor_ones(1,(size_t[]){4});
        pg_tensor *b = pg_tensor_zeros(1,(size_t[]){4});
        printf("x [2,4]:\n"); pg_tensor_print(x,stdout);
        pg_tensor *ln = pg_layernorm(x,w,b,1e-5f);
        printf("layernorm eps1e-5:\n"); pg_tensor_print(ln,stdout);
        pg_tensor *rn = pg_rmsnorm(x,w,1e-5f);
        printf("rmsnorm:\n"); pg_tensor_print(rn,stdout);
        pg_tensor_free(x); pg_tensor_free(w); pg_tensor_free(b);
        pg_tensor_free(ln); pg_tensor_free(rn);
    }

    // 8. indexing
    print_hdr("8. indexing (gather / scatter / index_select / masked)");
    {
        pg_tensor *t = pg_tensor_from_data(2,(size_t[]){2,3}, (float[]){10,20,30,40,50,60});
        printf("t [2,3]:\n"); pg_tensor_print(t,stdout);

        pg_tensor *idx = pg_tensor_from_data(1,(size_t[]){2}, (float[]){1,0});
        pg_tensor *sel = pg_index_select(t,0,idx);
        printf("index_select axis=0 [1,0]:\n"); pg_tensor_print(sel,stdout);

        pg_tensor *gi = pg_tensor_from_data(2,(size_t[]){2,2}, (float[]){0,2,1,1});
        pg_tensor *g = pg_gather(t,1,gi);
        printf("gather axis=1 indices [[0,2],[1,1]]:\n"); pg_tensor_print(g,stdout);

        pg_tensor *sv = pg_tensor_from_data(2,(size_t[]){2,2}, (float[]){-1,-2,-3,-4});
        pg_tensor *sc = pg_scatter(t,1,gi,sv);
        printf("scatter with same indices:\n"); pg_tensor_print(sc,stdout);

        pg_tensor *mask = pg_tensor_from_data(2,(size_t[]){2,3}, (float[]){1,0,1,0,1,0});
        pg_tensor *ms = pg_masked_select(t,mask);
        printf("masked_select [1,0,1,0,1,0]: "); pg_tensor_print(ms,stdout);

        pg_tensor_free(t); pg_tensor_free(idx); pg_tensor_free(sel);
        pg_tensor_free(gi); pg_tensor_free(g); pg_tensor_free(sv); pg_tensor_free(sc);
        pg_tensor_free(mask); pg_tensor_free(ms);
    }

    // 9. scan: cumsum/cumprod/sort/topk
    print_hdr("9. scan (cumsum / cumprod / sort / topk)");
    {
        pg_tensor *a = pg_tensor_from_data(2,(size_t[]){2,3}, (float[]){1,2,3,4,5,6});
        pg_tensor *cs = pg_cumsum(a,1);
        printf("cumsum axis=1:\n"); pg_tensor_print(cs,stdout);
        pg_tensor *cp = pg_cumprod(a,1);
        printf("cumprod axis=1:\n"); pg_tensor_print(cp,stdout);
        pg_tensor_free(cs); pg_tensor_free(cp); pg_tensor_free(a);

        float rraw[5]={3,-1,4,-1,5};
        pg_tensor *r = pg_tensor_from_data(2,(size_t[]){1,5}, rraw);
        pg_kv s = pg_sort(r,1,false);
        printf("sort asc values: "); pg_tensor_print(s.values,stdout);
        printf("       indices: "); pg_tensor_print(s.indices,stdout);
        pg_kv tk = pg_topk(r,1,2);
        printf("topk k=2 values: "); pg_tensor_print(tk.values,stdout);
        printf("         indices: "); pg_tensor_print(tk.indices,stdout);
        pg_tensor_free(r); pg_tensor_free(s.values); pg_tensor_free(s.indices);
        pg_tensor_free(tk.values); pg_tensor_free(tk.indices);
    }

    // 10. tiny JIT fusion demo (fits ops_showcase)
    print_hdr("10. JIT fusion (elementwise)");
    {
        pg_seed(7);
        pg_jit_graph *g = pg_jit_graph_new();
        int a = pg_jit_add_input(g,2,(size_t[]){4,8});
        int b = pg_jit_add_input(g,1,(size_t[]){8});
        int t0 = pg_jit_add(g,a,b); // broadcast
        int t1 = pg_jit_mul(g,t0,t0);
        int out = pg_jit_tanh(g, pg_jit_relu(g,t1));
        pg_jit_mark_output(g,out);
        pg_jit_exe *exe = pg_jit_compile(g);
        if(!exe){
            printf("JIT compile failed: %s\n", pg_jit_last_error());
        } else {
            pg_tensor *pa = pg_tensor_uniform(2,(size_t[]){4,8}, -1,1);
            pg_tensor *pb = pg_tensor_uniform(1,(size_t[]){8}, -1,1);
            const pg_tensor *ins[2]={pa,pb};
            pg_tensor *jout = pg_jit_run_single(exe, ins,2);

            // eager reference: tmp = (a+b); tmp2 = tmp*tmp; relu; tanh
            pg_tensor *tmp = pg_add(pa,pb);
            pg_tensor *tmp2 = pg_mul(tmp,tmp); pg_tensor_free(tmp);
            pg_tensor *re = pg_relu(tmp2); pg_tensor_free(tmp2);
            pg_tensor *ref = pg_tanh(re); pg_tensor_free(re);

            printf("JIT vs eager allclose: %s\n", pg_tensor_allclose(jout,ref,1e-4f,1e-4f)?"PASS":"FAIL");
            printf("JIT output [4,8] first row: ");
            for(int i=0;i<4;i++) printf("%g ", jout->data[i]);
            printf("\n");

            pg_tensor_free(pa); pg_tensor_free(pb); pg_tensor_free(jout); pg_tensor_free(ref);
            pg_jit_exe_free(exe);
        }
        pg_jit_graph_free(g);
        pg_jit_cache_clear();
    }

    printf("\npool: %zu entries, %zu bytes\n", pg_tensor_pool_size(), pg_tensor_pool_bytes());
    printf("\nops showcase finished.\n");
    return 0;
}
