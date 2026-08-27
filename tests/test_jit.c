#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/core/tensor.h"
#include "../src/jit/jit.h"
#include "../src/ops/activations.h"
#include "../src/ops/elementwise.h"

static int fails = 0;

#define CHECK(c) do{ if(!(c)){ printf("FAIL:%d %s\n", __LINE__, #c); fails++; } }while(0)
#define CHECKF(t,i,v) do{ if(!closef((t)->data[i], (v))){ printf("FAIL:%d %s[%zu]=%f want %f\n", __LINE__, #t, (size_t)(i), (t)->data[i], (double)(v)); fails++; } }while(0)

static bool closef(float a, float b){ return fabsf(a-b) <= 1e-4f*(1+fabsf(b)); }

static void test_basic_fusion(void){
    pg_jit_graph *g = pg_jit_graph_new();
    CHECK(g);
    int a = pg_jit_add_input(g, 2, (size_t[]){2,3});
    int b = pg_jit_add_input(g, 2, (size_t[]){2,3});
    int c = pg_jit_add_input(g, 2, (size_t[]){2,3});
    int t0 = pg_jit_add(g,a,b);
    int t1 = pg_jit_mul(g,t0,c);
    int t2 = pg_jit_relu(g,t1);
    pg_jit_mark_output(g,t2);
    pg_jit_exe *exe = pg_jit_compile(g);
    CHECK(exe);
    if(!exe){ pg_jit_graph_free(g); return; }
    float ad[6]={1,-2,3,4,-5,6};
    float bd[6]={6,5,4,3,2,1};
    float cd[6]={1,1,1,1,1,1};
    pg_tensor *ta=pg_tensor_from_data(2,(size_t[]){2,3},ad);
    pg_tensor *tb=pg_tensor_from_data(2,(size_t[]){2,3},bd);
    pg_tensor *tc=pg_tensor_from_data(2,(size_t[]){2,3},cd);
    const pg_tensor *ins[3]={ta,tb,tc};
    pg_tensor *out=pg_jit_run_single(exe,ins,3);
    CHECK(out);
    float expv[6]={7,3,7,7,0,7};
    for(size_t i=0;i<6;i++) CHECKF(out,i,expv[i]);
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc); pg_tensor_free(out);
    pg_jit_exe_free(exe); pg_jit_graph_free(g);
}

static void test_broadcast(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int in0=pg_jit_add_input(g,2,(size_t[]){2,4});
    int in1=pg_jit_add_input(g,1,(size_t[]){4});
    int t=pg_jit_add(g,in0,in1);
    int r=pg_jit_relu(g,t);
    pg_jit_mark_output(g,r);
    pg_jit_exe *e=pg_jit_compile(g);
    CHECK(e);
    if(!e){ pg_jit_graph_free(g); return; }
    float a0[8]={-1,2,-3,4,-5,6,-7,8};
    float b0[4]={10,20,30,40};
    pg_tensor *pa=pg_tensor_from_data(2,(size_t[]){2,4},a0);
    pg_tensor *pb=pg_tensor_from_data(1,(size_t[]){4},b0);
    const pg_tensor *ins[2]={pa,pb};
    pg_tensor *out=pg_jit_run_single(e,ins,2);
    CHECK(out);
    float exp2[8]={9,22,27,44,5,26,23,48};
    for(size_t i=0;i<8;i++) CHECKF(out,i,exp2[i]);
    // also compare with eager
    pg_tensor *eager=pg_add(pa,pb);
    pg_tensor *eager2=pg_relu(eager);
    for(size_t i=0;i<8;i++) CHECK(closef(out->data[i], eager2->data[i]));
    pg_tensor_free(pa); pg_tensor_free(pb); pg_tensor_free(out); pg_tensor_free(eager); pg_tensor_free(eager2);
    pg_jit_exe_free(e); pg_jit_graph_free(g);
}

static void test_3d_broadcast(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,3,(size_t[]){2,3,4});
    int b=pg_jit_add_input(g,2,(size_t[]){3,4});
    int c=pg_jit_add_input(g,1,(size_t[]){4});
    int t0=pg_jit_add(g,a,b); // a[2,3,4]+b[3,4] -> [2,3,4]
    int t1=pg_jit_mul(g,t0,c); // *c[4] -> [2,3,4]
    pg_jit_mark_output(g,t1);
    pg_jit_exe *e=pg_jit_compile(g);
    CHECK(e);
    if(!e){ pg_jit_graph_free(g); return; }
    pg_seed(99);
    pg_tensor *ta=pg_tensor_uniform(3,(size_t[]){2,3,4},-2,2);
    pg_tensor *tb=pg_tensor_uniform(2,(size_t[]){3,4},-2,2);
    pg_tensor *tc=pg_tensor_uniform(1,(size_t[]){4},-2,2);
    const pg_tensor *ins[3]={ta,tb,tc};
    pg_tensor *jout=pg_jit_run_single(e,ins,3);
    CHECK(jout);
    pg_tensor *e0=pg_add(ta,tb);
    pg_tensor *e1=pg_mul(e0,tc);
    for(size_t i=0;i<jout->numel;i++) CHECK(closef(jout->data[i], e1->data[i]));
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc);
    pg_tensor_free(jout); pg_tensor_free(e0); pg_tensor_free(e1);
    pg_jit_exe_free(e); pg_jit_graph_free(g);
}

static void test_unary_ops(void){
    pg_jit_op_t ops[]={PG_JIT_NEG,PG_JIT_EXP,PG_JIT_LOG,PG_JIT_SQRT,PG_JIT_SIN,PG_JIT_COS,PG_JIT_ABS,PG_JIT_RELU,PG_JIT_SIGMOID,PG_JIT_TANH,PG_JIT_GELU,PG_JIT_ERF};
    for(size_t oi=0; oi<sizeof(ops)/sizeof(ops[0]); ++oi){
        pg_jit_graph *g=pg_jit_graph_new();
        int inp=pg_jit_add_input(g,1,(size_t[]){8});
        int out=pg_jit_add_op(g,ops[oi],&inp,1);
        pg_jit_mark_output(g,out);
        pg_jit_exe *exe=pg_jit_compile(g);
        CHECK(exe);
        if(!exe){ pg_jit_graph_free(g); continue; }
        pg_tensor *t;
        if(ops[oi]==PG_JIT_LOG || ops[oi]==PG_JIT_SQRT) t=pg_tensor_uniform(1,(size_t[]){8},0.5f,2.0f);
        else t=pg_tensor_uniform(1,(size_t[]){8},-2.0f,2.0f);
        const pg_tensor *ins[1]={t};
        pg_tensor *jout=pg_jit_run_single(exe,ins,1);
        CHECK(jout);
        pg_tensor *eager=NULL;
        switch(ops[oi]){
            case PG_JIT_NEG: eager=pg_neg(t); break;
            case PG_JIT_EXP: eager=pg_exp(t); break;
            case PG_JIT_LOG: eager=pg_log(t); break;
            case PG_JIT_SQRT: eager=pg_sqrt(t); break;
            case PG_JIT_SIN: eager=pg_sin(t); break;
            case PG_JIT_COS: eager=pg_cos(t); break;
            case PG_JIT_ABS: eager=pg_abs(t); break;
            case PG_JIT_RELU: eager=pg_relu(t); break;
            case PG_JIT_SIGMOID: eager=pg_sigmoid(t); break;
            case PG_JIT_TANH: eager=pg_tanh(t); break;
            case PG_JIT_GELU: eager=pg_gelu(t); break;
            case PG_JIT_ERF: eager=pg_erf(t); break;
            default: break;
        }
        CHECK(eager);
        if(jout && eager){
            for(size_t i=0;i<8;i++) if(!closef(jout->data[i], eager->data[i])){ printf("FAIL unary op %d [%zu] jit %g eager %g\n", ops[oi], i, jout->data[i], eager->data[i]); fails++; break; }
        }
        pg_tensor_free(t); pg_tensor_free(jout); pg_tensor_free(eager);
        pg_jit_exe_free(exe); pg_jit_graph_free(g);
    }
}

static void test_binary_ops(void){
    pg_jit_op_t ops[]={PG_JIT_ADD,PG_JIT_SUB,PG_JIT_MUL,PG_JIT_DIV};
    for(size_t oi=0; oi<sizeof(ops)/sizeof(ops[0]); ++oi){
        pg_jit_graph *g=pg_jit_graph_new();
        int a=pg_jit_add_input(g,2,(size_t[]){2,3});
        int b=pg_jit_add_input(g,2,(size_t[]){2,3});
        int out=pg_jit_add_op(g,ops[oi],(int[]){a,b},2);
        pg_jit_mark_output(g,out);
        pg_jit_exe *exe=pg_jit_compile(g);
        CHECK(exe);
        if(!exe){ pg_jit_graph_free(g); continue; }
        pg_tensor *ta=pg_tensor_uniform(2,(size_t[]){2,3},-2,2);
        pg_tensor *tb=pg_tensor_uniform(2,(size_t[]){2,3},0.5,2);
        const pg_tensor *ins[2]={ta,tb};
        pg_tensor *jout=pg_jit_run_single(exe,ins,2);
        pg_tensor *eager=NULL;
        switch(ops[oi]){
            case PG_JIT_ADD: eager=pg_add(ta,tb); break;
            case PG_JIT_SUB: eager=pg_sub(ta,tb); break;
            case PG_JIT_MUL: eager=pg_mul(ta,tb); break;
            case PG_JIT_DIV: eager=pg_div(ta,tb); break;
            default: break;
        }
        CHECK(jout && eager);
        if(jout && eager) for(size_t i=0;i<6;i++) CHECK(closef(jout->data[i], eager->data[i]));
        pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(jout); pg_tensor_free(eager);
        pg_jit_exe_free(exe); pg_jit_graph_free(g);
    }
}

static void test_const_broadcast(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,2,(size_t[]){2,4});
    int c=pg_jit_add_const(g,2.5f);
    int m=pg_jit_mul(g,a,c);
    int r=pg_jit_relu(g,m);
    pg_jit_mark_output(g,r);
    pg_jit_exe *exe=pg_jit_compile(g);
    CHECK(exe);
    if(!exe){ pg_jit_graph_free(g); return; }
    pg_tensor *ta=pg_tensor_uniform(2,(size_t[]){2,4},-2,2);
    const pg_tensor *ins[1]={ta};
    pg_tensor *jout=pg_jit_run_single(exe,ins,1);
    CHECK(jout);
    pg_tensor *cf=pg_tensor_full(2,(size_t[]){2,4},2.5f);
    pg_tensor *mul=pg_mul(ta,cf);
    pg_tensor *exp=pg_relu(mul);
    for(size_t i=0;i<8;i++) CHECK(closef(jout->data[i], exp->data[i]));
    pg_tensor_free(ta); pg_tensor_free(jout); pg_tensor_free(cf); pg_tensor_free(mul); pg_tensor_free(exp);
    pg_jit_exe_free(exe); pg_jit_graph_free(g);
}

static void test_complex_fused(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,2,(size_t[]){4,8});
    int b=pg_jit_add_input(g,1,(size_t[]){8});
    int c=pg_jit_add_input(g,2,(size_t[]){4,8});
    int d=pg_jit_add_const(g,0.5f);
    int t0=pg_jit_add(g,a,b);
    int t1=pg_jit_mul(g,t0,c);
    int t2=pg_jit_add(g,t1,d);
    int t3=pg_jit_tanh(g,t2);
    pg_jit_mark_output(g,t3);
    pg_jit_exe *exe=pg_jit_compile(g);
    CHECK(exe);
    if(!exe){ pg_jit_graph_free(g); return; }
    pg_seed(123);
    pg_tensor *ta=pg_tensor_uniform(2,(size_t[]){4,8},-1,1);
    pg_tensor *tb=pg_tensor_uniform(1,(size_t[]){8},-0.5,0.5);
    pg_tensor *tc=pg_tensor_uniform(2,(size_t[]){4,8},-1,1);
    const pg_tensor *ins[3]={ta,tb,tc};
    pg_tensor *jout=pg_jit_run_single(exe,ins,3);
    CHECK(jout);
    pg_tensor *e_add=pg_add(ta,tb);
    pg_tensor *e_mul=pg_mul(e_add,tc);
    pg_tensor *cf=pg_tensor_full(2,(size_t[]){4,8},0.5f);
    pg_tensor *e_add2=pg_add(e_mul,cf);
    pg_tensor *e_tanh=pg_tanh(e_add2);
    for(size_t i=0;i<e_tanh->numel;i++) CHECK(closef(jout->data[i], e_tanh->data[i]));
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc); pg_tensor_free(jout);
    pg_tensor_free(e_add); pg_tensor_free(e_mul); pg_tensor_free(cf); pg_tensor_free(e_add2); pg_tensor_free(e_tanh);
    pg_jit_exe_free(exe); pg_jit_graph_free(g);
}

static void test_multi_output(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,1,(size_t[]){4});
    int b=pg_jit_add_input(g,1,(size_t[]){4});
    int add=pg_jit_add(g,a,b);
    int mul=pg_jit_mul(g,a,b);
    pg_jit_mark_output(g,add);
    pg_jit_mark_output(g,mul);
    pg_jit_exe *exe=pg_jit_compile(g);
    CHECK(exe);
    if(!exe){ pg_jit_graph_free(g); return; }
    pg_tensor *ta=pg_tensor_from_data(1,(size_t[]){4},(float[]){1,2,3,4});
    pg_tensor *tb=pg_tensor_from_data(1,(size_t[]){4},(float[]){5,6,7,8});
    const pg_tensor *ins[2]={ta,tb};
    pg_tensor *out_add=pg_tensor_zeros(1,(size_t[]){4});
    pg_tensor *out_mul=pg_tensor_zeros(1,(size_t[]){4});
    pg_tensor *outs[2]={out_add,out_mul};
    bool ok=pg_jit_run(exe,ins,2,outs,2);
    CHECK(ok);
    float exp_add[4]={6,8,10,12};
    float exp_mul[4]={5,12,21,32};
    for(size_t i=0;i<4;i++){ CHECK(closef(out_add->data[i], exp_add[i])); CHECK(closef(out_mul->data[i], exp_mul[i])); }
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(out_add); pg_tensor_free(out_mul);
    pg_jit_exe_free(exe); pg_jit_graph_free(g);
}

static void test_cache(void){
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,1,(size_t[]){4});
    int b=pg_jit_add_input(g,1,(size_t[]){4});
    int c=pg_jit_add(g,a,b);
    pg_jit_mark_output(g,c);
    size_t before=pg_jit_cache_size();
    pg_jit_exe *e1=pg_jit_compile(g);
    CHECK(e1);
    size_t mid=pg_jit_cache_size();
    CHECK(mid==before+1);
    pg_jit_exe *e2=pg_jit_compile(g);
    CHECK(e2);
    size_t after=pg_jit_cache_size();
    CHECK(after==mid);
    // hash consistency
    CHECK(pg_jit_graph_hash(g)==pg_jit_graph_hash(g));
    pg_jit_exe_free(e1); pg_jit_exe_free(e2);
    pg_jit_graph_free(g);
}

static void test_error_handling(void){
    pg_jit_graph *g=pg_jit_graph_new();
    // no outputs -> compile should fail
    pg_jit_exe *e=pg_jit_compile(g);
    CHECK(e==NULL);
    // invalid ids
    int bad=pg_jit_add(g, 999, 1000);
    CHECK(bad==-1);
    pg_jit_graph_free(g);
    // shape mismatch at run
    pg_jit_graph *g2=pg_jit_graph_new();
    int a=pg_jit_add_input(g2,1,(size_t[]){4});
    pg_jit_mark_output(g2,a);
    pg_jit_exe *e2=pg_jit_compile(g2);
    CHECK(e2);
    if(e2){
        pg_tensor *ta=pg_tensor_zeros(1,(size_t[]){5}); // wrong shape
        pg_tensor *out=pg_tensor_zeros(1,(size_t[]){4});
        const pg_tensor *ins[1]={ta};
        pg_tensor *outs[1]={out};
        bool ok=pg_jit_run(e2,ins,1,outs,1);
        CHECK(!ok);
        pg_tensor_free(ta); pg_tensor_free(out);
        pg_jit_exe_free(e2);
    }
    pg_jit_graph_free(g2);
}

static void test_large_fusion_perf(void){
    // sanity: large tensor fused vs eager result still matches
    pg_jit_graph *g=pg_jit_graph_new();
    int a=pg_jit_add_input(g,2,(size_t[]){64,64});
    int b=pg_jit_add_input(g,2,(size_t[]){64,64});
    int c=pg_jit_add_input(g,2,(size_t[]){64,64});
    int t0=pg_jit_add(g,a,b);
    int t1=pg_jit_mul(g,t0,c);
    int t2=pg_jit_tanh(g,t1);
    int t3=pg_jit_relu(g,t2);
    pg_jit_mark_output(g,t3);
    pg_jit_exe *exe=pg_jit_compile(g);
    CHECK(exe);
    if(!exe){ pg_jit_graph_free(g); return; }
    pg_seed(77);
    pg_tensor *ta=pg_tensor_uniform(2,(size_t[]){64,64},-1,1);
    pg_tensor *tb=pg_tensor_uniform(2,(size_t[]){64,64},-1,1);
    pg_tensor *tc=pg_tensor_uniform(2,(size_t[]){64,64},-1,1);
    const pg_tensor *ins[3]={ta,tb,tc};
    pg_tensor *jout=pg_jit_run_single(exe,ins,3);
    CHECK(jout);
    pg_tensor *e0=pg_add(ta,tb);
    pg_tensor *e1=pg_mul(e0,tc);
    pg_tensor *e2=pg_tanh(e1);
    pg_tensor *e3=pg_relu(e2);
    for(size_t i=0;i<jout->numel;i++) CHECK(closef(jout->data[i], e3->data[i]));
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc);
    pg_tensor_free(jout); pg_tensor_free(e0); pg_tensor_free(e1); pg_tensor_free(e2); pg_tensor_free(e3);
    pg_jit_exe_free(exe); pg_jit_graph_free(g);
}

int main(void){
    test_basic_fusion();
    test_broadcast();
    test_3d_broadcast();
    test_unary_ops();
    test_binary_ops();
    test_const_broadcast();
    test_complex_fused();
    test_multi_output();
    test_cache();
    test_error_handling();
    test_large_fusion_perf();
    pg_jit_cache_clear();
    if(fails==0) printf("test_jit: all passed\n");
    else printf("test_jit: %d failures\n", fails);
    return fails!=0;
}
