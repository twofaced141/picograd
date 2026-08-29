#include <stdio.h>
#include <time.h>

#include "../src/core/tensor.h"
#include "../src/jit/jit.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/activations.h"

/*
 * JIT demo: fuse a chain of elementwise ops into a single compiled kernel.
 * Demonstrates:
 *  - graph building
 *  - compilation (C code -> shared lib via cc)
 *  - caching
 *  - execution and verification against eager
 *  - speedup measurement
 */

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec/1e6;
}

int main(void){
    printf("=== picograd JIT demo ===\n\n");

    /* 1. Build graph: (a + b) * c -> relu -> tanh
     *    All tensors [64,64], no broadcasting.
     */
    pg_jit_graph *g = pg_jit_graph_new();
    int a = pg_jit_add_input(g, 2, (size_t[]){64, 64});
    int b = pg_jit_add_input(g, 2, (size_t[]){64, 64});
    int c = pg_jit_add_input(g, 2, (size_t[]){64, 64});
    int t0 = pg_jit_add(g, a, b);
    int t1 = pg_jit_mul(g, t0, c);
    int t2 = pg_jit_relu(g, t1);
    int t3 = pg_jit_tanh(g, t2);
    pg_jit_mark_output(g, t3);

    printf("Graph: (a + b) * c -> relu -> tanh  (%zu nodes)\n", pg_jit_num_nodes(g));

    /* 2. Compilation */
    double t_compile_start = now_ms();
    pg_jit_exe *exe = pg_jit_compile(g);
    double t_compile = now_ms() - t_compile_start;
    if(!exe){
        printf("JIT compile failed: %s\n", pg_jit_last_error());
        return 1;
    }
    printf("Compiled in %.1f ms\n", t_compile);
    printf("Cache size: %zu, hash=0x%llx\n", pg_jit_cache_size(), (unsigned long long)pg_jit_graph_hash(g));

    /* recompile the same graph — cache hit, no recompilation */
    pg_jit_exe *exe2 = pg_jit_compile(g);
    printf("Recompile (cache hit): %s\n", pg_jit_cache_size()==1 ? "OK" : "FAIL");
    pg_jit_exe_free(exe2);

    /* 3. Prepare data */
    pg_seed(42);
    pg_tensor *ta = pg_tensor_uniform(2,(size_t[]){64,64}, -1.0f, 1.0f);
    pg_tensor *tb = pg_tensor_uniform(2,(size_t[]){64,64}, -1.0f, 1.0f);
    pg_tensor *tc = pg_tensor_uniform(2,(size_t[]){64,64}, -1.0f, 1.0f);
    const pg_tensor *ins[3]={ta,tb,tc};

    /* eager for verification */
    pg_tensor *e0 = pg_add(ta,tb);
    pg_tensor *e1 = pg_mul(e0,tc);
    pg_tensor *e2 = pg_relu(e1);
    pg_tensor *eager = pg_tanh(e2);

    pg_tensor *jout = pg_jit_run_single(exe, ins, 3);
    bool ok = pg_tensor_allclose(jout, eager, 1e-4f, 1e-4f);
    printf("Verification vs eager: %s\n", ok ? "PASS" : "FAIL");

    /* 4. Benchmark: eager 4 memory passes vs JIT 1 pass */
    const int ITERS = 10000;
    double t0_ms = now_ms();
    for(int i=0;i<ITERS;i++){
        pg_tensor *x0 = pg_add(ta,tb);
        pg_tensor *x1 = pg_mul(x0,tc);
        pg_tensor *x2 = pg_relu(x1);
        pg_tensor *x3 = pg_tanh(x2);
        pg_tensor_free(x0); pg_tensor_free(x1); pg_tensor_free(x2); pg_tensor_free(x3);
    }
    double t_eager = now_ms() - t0_ms;

    double t1_ms = now_ms();
    for(int i=0;i<ITERS;i++){
        pg_tensor *y = pg_jit_run_single(exe, ins, 3);
        pg_tensor_free(y);
    }
    double t_jit = now_ms() - t1_ms;

    printf("\nBenchmark %d iterations [64,64]:\n", ITERS);
    printf("  eager: %.1f ms\n", t_eager);
    printf("  jit  : %.1f ms  (%.1fx)\n", t_jit, t_eager / (t_jit>0?t_jit:1));

    /* 5. Broadcast: [4,8] + [8] -> relu (common bias case) */
    printf("\n--- broadcast [4,8] + [8] ---\n");
    pg_jit_graph *gb = pg_jit_graph_new();
    int ba = pg_jit_add_input(gb, 2, (size_t[]){4,8});
    int bb = pg_jit_add_input(gb, 1, (size_t[]){8});
    int br = pg_jit_relu(gb, pg_jit_add(gb, ba, bb));
    pg_jit_mark_output(gb, br);
    pg_jit_exe *exeb = pg_jit_compile(gb);
    pg_tensor *pa = pg_tensor_uniform(2,(size_t[]){4,8},-1,1);
    pg_tensor *pb = pg_tensor_uniform(1,(size_t[]){8},-1,1);
    const pg_tensor *insb[2]={pa,pb};
    pg_tensor *jb = pg_jit_run_single(exeb, insb, 2);
    pg_tensor *tmp = pg_add(pa,pb);
    pg_tensor *eb2 = pg_relu(tmp);
    printf("broadcast verification: %s\n", pg_tensor_allclose(jb, eb2,1e-5,1e-5)?"PASS":"FAIL");

    /* cleanup */
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc);
    pg_tensor_free(e0); pg_tensor_free(e1); pg_tensor_free(e2); pg_tensor_free(eager); pg_tensor_free(jout);
    pg_tensor_free(pa); pg_tensor_free(pb); pg_tensor_free(jb); pg_tensor_free(tmp); pg_tensor_free(eb2);
    pg_jit_exe_free(exe); pg_jit_exe_free(exeb);
    pg_jit_graph_free(g); pg_jit_graph_free(gb);
    pg_jit_cache_clear();

    printf("\nJIT demo finished.\n");
    return 0;
}
