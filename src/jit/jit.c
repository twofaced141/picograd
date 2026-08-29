#include "jit.h"

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---------- internal structures ---------- */

typedef enum { JNODE_INPUT, JNODE_CONST, JNODE_OP } jnode_kind_t;

typedef struct {
    jnode_kind_t kind;
    pg_jit_op_t op; /* valid if kind==JNODE_OP */
    int inputs[2];
    size_t ninputs;
    float const_val; /* if CONST */
    size_t ndim;
    size_t shape[PG_MAX_NDIM];
    size_t numel;
    size_t stride[PG_MAX_NDIM];
} jnode_t;

struct pg_jit_graph {
    jnode_t *nodes;
    size_t nnodes, cap;
    int *inputs; /* list of INPUT node ids in order */
    size_t ninputs, cap_inputs;
    int *outputs;
    size_t noutputs, cap_outputs;
};

struct pg_jit_exe {
    void (*kernel)(float **, const float **);
    void *handle;
    char c_path[512];
    char so_path[512];
    size_t ninputs;
    size_t noutputs;
    size_t in_ndim[16];
    size_t in_shape[16][PG_MAX_NDIM];
    size_t out_ndim[16];
    size_t out_shape[16][PG_MAX_NDIM];
    size_t out_numel[16];
    uint64_t hash;
};

/* ---------- cache ---------- */
typedef struct cache_entry {
    uint64_t hash;
    char so_path[512];
    char c_path[512];
    size_t ninputs, noutputs;
    size_t in_ndim[16];
    size_t in_shape[16][PG_MAX_NDIM];
    size_t out_ndim[16];
    size_t out_shape[16][PG_MAX_NDIM];
    size_t out_numel[16];
    /* keep copy of graph for equality check */
    pg_jit_graph *graph_copy;
    struct cache_entry *next;
} cache_entry_t;

static cache_entry_t *g_cache_head = NULL;
static size_t g_cache_size = 0;
static bool g_cache_enabled = true;
static char g_last_err[1024] = {0};
static size_t g_jit_counter = 0;

static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_err, sizeof(g_last_err), fmt, ap);
    va_end(ap);
}

const char *pg_jit_last_error(void) { return g_last_err; }
size_t pg_jit_cache_size(void) { return g_cache_size; }
void pg_jit_set_cache_enabled(bool e) { g_cache_enabled = e; }
bool pg_jit_is_cache_enabled(void) { return g_cache_enabled; }

/* ---------- helpers ---------- */
static bool valid_shape(size_t ndim, const size_t *shape) {
    if (ndim == 0 || ndim > PG_MAX_NDIM || !shape) return false;
    for (size_t i = 0; i < ndim; i++) if (shape[i]==0) return false;
    return true;
}
static size_t shape_numel(size_t ndim, const size_t *shape) {
    size_t n=1;
    for (size_t i=0;i<ndim;i++) n*=shape[i];
    return n;
}
static void compute_strides(size_t ndim, const size_t *shape, size_t *stride) {
    size_t acc=1;
    for (size_t i=ndim; i-- >0;) { stride[i]=acc; acc*=shape[i]; }
}
static bool shape_equal(size_t na, const size_t *sa, size_t nb, const size_t *sb) {
    if (na!=nb) return false;
    for (size_t i=0;i<na;i++) if (sa[i]!=sb[i]) return false;
    return true;
}

/* broadcast two shapes -> out_shape/nDim . Returns false if incompatible. */
static bool bcast_shape2(size_t na, const size_t *sa, size_t nb, const size_t *sb,
                         size_t *out_ndim, size_t *out_shape) {
    size_t nd = na > nb ? na : nb;
    if (nd > PG_MAX_NDIM) return false;
    for (size_t d=0; d<nd; d++) {
        size_t da = d < nd - na ? 1 : sa[d - (nd - na)];
        size_t db = d < nd - nb ? 1 : sb[d - (nd - nb)];
        if (da!=db && da!=1 && db!=1) return false;
        out_shape[d] = da > db ? da : db;
    }
    *out_ndim = nd;
    return true;
}

/* for unary, out = in */
static int graph_ensure_cap(pg_jit_graph *g) {
    if (g->nnodes < g->cap) return 0;
    size_t nc = g->cap ? g->cap*2 : 16;
    jnode_t *nn = realloc(g->nodes, nc*sizeof(*nn));
    if (!nn) return -1;
    g->nodes = nn; g->cap = nc; return 0;
}
static int inputs_ensure_cap(pg_jit_graph *g) {
    if (g->ninputs < g->cap_inputs) return 0;
    size_t nc = g->cap_inputs ? g->cap_inputs*2 : 8;
    int *nn = realloc(g->inputs, nc*sizeof(*nn));
    if (!nn) return -1;
    g->inputs = nn; g->cap_inputs = nc; return 0;
}
static int outputs_ensure_cap(pg_jit_graph *g) {
    if (g->noutputs < g->cap_outputs) return 0;
    size_t nc = g->cap_outputs ? g->cap_outputs*2 : 4;
    int *nn = realloc(g->outputs, nc*sizeof(*nn));
    if (!nn) return -1;
    g->outputs = nn; g->cap_outputs = nc; return 0;
}

static uint64_t fnv1a_64(const void *data, size_t len, uint64_t h) {
    const uint8_t *p=(const uint8_t*)data;
    for (size_t i=0;i<len;i++){ h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---------- graph API ---------- */
pg_jit_graph *pg_jit_graph_new(void) {
    pg_jit_graph *g = calloc(1, sizeof(*g));
    return g;
}
void pg_jit_graph_free(pg_jit_graph *g) {
    if (!g) return;
    free(g->nodes);
    free(g->inputs);
    free(g->outputs);
    free(g);
}
size_t pg_jit_num_inputs(const pg_jit_graph *g){ return g?g->ninputs:0; }
size_t pg_jit_num_outputs(const pg_jit_graph *g){ return g?g->noutputs:0; }
size_t pg_jit_num_nodes(const pg_jit_graph *g){ return g?g->nnodes:0; }

int pg_jit_add_input(pg_jit_graph *g, size_t ndim, const size_t *shape) {
    if (!g || !valid_shape(ndim, shape)) { set_err("invalid input shape"); return -1; }
    if (graph_ensure_cap(g)!=0 || inputs_ensure_cap(g)!=0) { set_err("oom"); return -1; }
    jnode_t *n = &g->nodes[g->nnodes];
    memset(n,0,sizeof(*n));
    n->kind = JNODE_INPUT;
    n->ndim = ndim;
    memcpy(n->shape, shape, ndim*sizeof(size_t));
    n->numel = shape_numel(ndim, shape);
    compute_strides(ndim, shape, n->stride);
    int id = (int)g->nnodes;
    g->nnodes++;
    g->inputs[g->ninputs++] = id;
    return id;
}
int pg_jit_add_const(pg_jit_graph *g, float value) {
    if (!g) { set_err("null graph"); return -1; }
    if (graph_ensure_cap(g)!=0) { set_err("oom"); return -1; }
    jnode_t *n = &g->nodes[g->nnodes];
    memset(n,0,sizeof(*n));
    n->kind = JNODE_CONST;
    n->const_val = value;
    n->ndim = 1; n->shape[0]=1; n->numel=1; n->stride[0]=1;
    int id = (int)g->nnodes;
    g->nnodes++;
    return id;
}

static bool op_is_binary(pg_jit_op_t op){
    return op==PG_JIT_ADD || op==PG_JIT_SUB || op==PG_JIT_MUL || op==PG_JIT_DIV;
}
static bool op_is_unary(pg_jit_op_t op){
    return !op_is_binary(op);
}

int pg_jit_add_op(pg_jit_graph *g, pg_jit_op_t op, const int *inputs, size_t ninputs) {
    if (!g || !inputs) { set_err("null args"); return -1; }
    bool is_bin = op_is_binary(op);
    bool is_una = op_is_unary(op);
    if (is_bin && ninputs!=2) { set_err("binary op requires 2 inputs"); return -1; }
    if (is_una && ninputs!=1) { set_err("unary op requires 1 input"); return -1; }
    for (size_t i=0;i<ninputs;i++) {
        if (inputs[i] <0 || (size_t)inputs[i] >= g->nnodes) { set_err("invalid input id %d", inputs[i]); return -1; }
    }
    // compute output shape
    size_t out_ndim=0; size_t out_shape[PG_MAX_NDIM]={0};
    if (is_bin) {
        jnode_t *a=&g->nodes[inputs[0]];
        jnode_t *b=&g->nodes[inputs[1]];
        // const nodes have shape [1] – broadcasting works
        if (!bcast_shape2(a->ndim, a->shape, b->ndim, b->shape, &out_ndim, out_shape)) {
            set_err("incompatible broadcast shapes"); return -1;
        }
    } else {
        jnode_t *a=&g->nodes[inputs[0]];
        out_ndim=a->ndim;
        memcpy(out_shape, a->shape, out_ndim*sizeof(size_t));
    }
    if (graph_ensure_cap(g)!=0) { set_err("oom"); return -1; }
    jnode_t *n=&g->nodes[g->nnodes];
    memset(n,0,sizeof(*n));
    n->kind=JNODE_OP; n->op=op; n->ninputs=ninputs;
    for (size_t i=0;i<ninputs;i++) n->inputs[i]=inputs[i];
    n->ndim=out_ndim; memcpy(n->shape,out_shape,out_ndim*sizeof(size_t));
    n->numel=shape_numel(out_ndim,out_shape);
    compute_strides(out_ndim,out_shape,n->stride);
    int id=(int)g->nnodes;
    g->nnodes++;
    return id;
}

/* wrappers */
int pg_jit_add(pg_jit_graph *g, int a, int b){ int ins[2]={a,b}; return pg_jit_add_op(g, PG_JIT_ADD, ins, 2); }
int pg_jit_sub(pg_jit_graph *g, int a, int b){ int ins[2]={a,b}; return pg_jit_add_op(g, PG_JIT_SUB, ins, 2); }
int pg_jit_mul(pg_jit_graph *g, int a, int b){ int ins[2]={a,b}; return pg_jit_add_op(g, PG_JIT_MUL, ins, 2); }
int pg_jit_div(pg_jit_graph *g, int a, int b){ int ins[2]={a,b}; return pg_jit_add_op(g, PG_JIT_DIV, ins, 2); }
int pg_jit_neg(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_NEG, &a, 1); }
int pg_jit_exp(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_EXP, &a, 1); }
int pg_jit_log(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_LOG, &a, 1); }
int pg_jit_sqrt(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_SQRT, &a, 1); }
int pg_jit_sin(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_SIN, &a, 1); }
int pg_jit_cos(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_COS, &a, 1); }
int pg_jit_abs(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_ABS, &a, 1); }
int pg_jit_relu(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_RELU, &a, 1); }
int pg_jit_sigmoid(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_SIGMOID, &a, 1); }
int pg_jit_tanh(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_TANH, &a, 1); }
int pg_jit_gelu(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_GELU, &a, 1); }
int pg_jit_erf(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_ERF, &a, 1); }
int pg_jit_step(pg_jit_graph *g, int a){ return pg_jit_add_op(g, PG_JIT_STEP, &a, 1); }

void pg_jit_mark_output(pg_jit_graph *g, int id){
    if (!g || id<0 || (size_t)id>=g->nnodes) { set_err("invalid output id"); return; }
    if (outputs_ensure_cap(g)!=0) { set_err("oom"); return; }
    g->outputs[g->noutputs++] = id;
}
void pg_jit_clear_outputs(pg_jit_graph *g){ if(g) g->noutputs=0; }

/* ---------- hash ---------- */
uint64_t pg_jit_graph_hash(const pg_jit_graph *g){
    uint64_t h = 14695981039346656037ULL;
    if (!g) return h;
    h = fnv1a_64(&g->nnodes, sizeof(g->nnodes), h);
    h = fnv1a_64(&g->ninputs, sizeof(g->ninputs), h);
    h = fnv1a_64(&g->noutputs, sizeof(g->noutputs), h);
    for (size_t i=0;i<g->nnodes;i++){
        const jnode_t *n=&g->nodes[i];
        h = fnv1a_64(&n->kind, sizeof(n->kind), h);
        if (n->kind==JNODE_OP) {
            h = fnv1a_64(&n->op, sizeof(n->op), h);
            h = fnv1a_64(n->inputs, n->ninputs*sizeof(int), h);
        } else if (n->kind==JNODE_CONST) {
            h = fnv1a_64(&n->const_val, sizeof(n->const_val), h);
        }
        h = fnv1a_64(&n->ndim, sizeof(n->ndim), h);
        h = fnv1a_64(n->shape, n->ndim*sizeof(size_t), h);
    }
    if (g->ninputs) h = fnv1a_64(g->inputs, g->ninputs*sizeof(int), h);
    if (g->noutputs) h = fnv1a_64(g->outputs, g->noutputs*sizeof(int), h);
    return h;
}
static bool graph_equal(const pg_jit_graph *a, const pg_jit_graph *b){
    if (!a || !b) return false;
    if (a->nnodes!=b->nnodes || a->ninputs!=b->ninputs || a->noutputs!=b->noutputs) return false;
    for (size_t i=0;i<a->nnodes;i++){
        const jnode_t *na=&a->nodes[i], *nb=&b->nodes[i];
        if (na->kind!=nb->kind || na->ndim!=nb->ndim || na->numel!=nb->numel) return false;
        if (na->kind==JNODE_OP && (na->op!=nb->op || na->ninputs!=nb->ninputs)) return false;
        if (na->kind==JNODE_OP) for (size_t k=0;k<na->ninputs;k++) if (na->inputs[k]!=nb->inputs[k]) return false;
        if (na->kind==JNODE_CONST && na->const_val!=nb->const_val) return false;
        for (size_t d=0; d<na->ndim; d++) if (na->shape[d]!=nb->shape[d]) return false;
    }
    for (size_t i=0;i<a->ninputs;i++) if (a->inputs[i]!=b->inputs[i]) return false;
    for (size_t i=0;i<a->noutputs;i++) if (a->outputs[i]!=b->outputs[i]) return false;
    return true;
}
static pg_jit_graph *graph_clone(const pg_jit_graph *g){
    if (!g) return NULL;
    pg_jit_graph *c = pg_jit_graph_new();
    if (!c) return NULL;
    c->nnodes=g->nnodes; c->ninputs=g->ninputs; c->noutputs=g->noutputs;
    c->cap=g->nnodes; c->cap_inputs=g->ninputs; c->cap_outputs=g->noutputs;
    if (g->nnodes){ c->nodes=malloc(c->cap*sizeof(jnode_t)); if(!c->nodes){free(c);return NULL;} memcpy(c->nodes,g->nodes,g->nnodes*sizeof(jnode_t)); }
    if (g->ninputs){ c->inputs=malloc(c->cap_inputs*sizeof(int)); if(!c->inputs){free(c->nodes);free(c);return NULL;} memcpy(c->inputs,g->inputs,g->ninputs*sizeof(int)); }
    if (g->noutputs){ c->outputs=malloc(c->cap_outputs*sizeof(int)); if(!c->outputs){free(c->nodes);free(c->inputs);free(c);return NULL;} memcpy(c->outputs,g->outputs,g->noutputs*sizeof(int)); }
    return c;
}

/* ---------- codegen helpers ---------- */
static const char *op_to_expr(pg_jit_op_t op, const char *a, const char *b, char *buf, size_t blen){
    switch(op){
        case PG_JIT_ADD: snprintf(buf,blen,"(%s + %s)",a,b); break;
        case PG_JIT_SUB: snprintf(buf,blen,"(%s - %s)",a,b); break;
        case PG_JIT_MUL: snprintf(buf,blen,"(%s * %s)",a,b); break;
        case PG_JIT_DIV: snprintf(buf,blen,"(%s / %s)",a,b); break;
        case PG_JIT_NEG: snprintf(buf,blen,"(-%s)",a); break;
        case PG_JIT_EXP: snprintf(buf,blen,"expf(%s)",a); break;
        case PG_JIT_LOG: snprintf(buf,blen,"logf(%s)",a); break;
        case PG_JIT_SQRT: snprintf(buf,blen,"sqrtf(%s)",a); break;
        case PG_JIT_SIN: snprintf(buf,blen,"sinf(%s)",a); break;
        case PG_JIT_COS: snprintf(buf,blen,"cosf(%s)",a); break;
        case PG_JIT_ABS: snprintf(buf,blen,"fabsf(%s)",a); break;
        case PG_JIT_RELU: snprintf(buf,blen,"((%s) > 0.0f ? (%s) : 0.0f)",a,a); break;
        case PG_JIT_SIGMOID: snprintf(buf,blen,"(1.0f / (1.0f + expf(-(%s))))",a); break;
        case PG_JIT_TANH: snprintf(buf,blen,"tanhf(%s)",a); break;
        case PG_JIT_GELU: snprintf(buf,blen,"(0.5f * (%s) * (1.0f + erff((%s) * 0.70710678118f)))",a,a); break;
        case PG_JIT_ERF: snprintf(buf,blen,"erff(%s)",a); break;
        case PG_JIT_STEP: snprintf(buf,blen,"((%s) > 0.0f ? 1.0f : 0.0f)",a); break;
        default: snprintf(buf,blen,"0.0f"); break;
    }
    return buf;
}

/* compute broadcast strides for input vs output shape.
   out_ndim/out_shape is loop shape. in_node has shape. fills bstride[ out_ndim ]. Returns true. */
static bool compute_bcast_strides(const jnode_t *in, size_t out_ndim, const size_t *out_shape, size_t *bstride){
    size_t in_stride[PG_MAX_NDIM]={0};
    compute_strides(in->ndim, in->shape, in_stride);
    size_t off = out_ndim > in->ndim ? out_ndim - in->ndim : 0;
    for (size_t d=0; d<out_ndim; d++){
        if (d < off) { bstride[d]=0; continue; }
        size_t id = d - off;
        if (in->shape[id]==1 && out_shape[d]!=1) bstride[d]=0;
        else if (in->shape[id]==out_shape[d]) bstride[d]=in_stride[id];
        else if (in->ndim==1 && in->shape[0]==1) bstride[d]=0; // scalar case
        else return false;
    }
    return true;
}

/* check if input shape equals output shape exactly */
static bool is_direct(const jnode_t *in, size_t out_ndim, const size_t *out_shape){
    if (in->ndim!=out_ndim) return false;
    return shape_equal(in->ndim,in->shape,out_ndim,out_shape);
}

/* reachable marking */
static void mark_reachable(const pg_jit_graph *g, bool *reach){
    memset(reach,0,g->nnodes);
    int *stack = malloc(g->nnodes*sizeof(int));
    size_t sp=0;
    for (size_t i=0;i<g->noutputs;i++){
        int oid=g->outputs[i];
        if (!reach[oid]){ reach[oid]=true; stack[sp++]=oid; }
    }
    while(sp){
        int cur=stack[--sp];
        const jnode_t *n=&g->nodes[cur];
        if (n->kind==JNODE_OP){
            for (size_t k=0;k<n->ninputs;k++){
                int inp=n->inputs[k];
                if (!reach[inp]){ reach[inp]=true; stack[sp++]=inp; }
            }
        }
    }
    free(stack);
}

/* emit C code */
static bool emit_c_code(const pg_jit_graph *g, FILE *f){
    if (!g || g->noutputs==0) { set_err("no outputs"); return false; }
    // all outputs must share the same shape
    size_t out0_id = g->outputs[0];
    const jnode_t *out0 = &g->nodes[out0_id];
    size_t out_ndim = out0->ndim;
    const size_t *out_shape = out0->shape;
    size_t out_numel = out0->numel;
    for (size_t i=1;i<g->noutputs;i++){
        const jnode_t *o=&g->nodes[g->outputs[i]];
        if (o->numel!=out_numel) { set_err("multi-output with different numel not supported (%zu vs %zu)",o->numel,out_numel); return false; }
        if (!shape_equal(o->ndim,o->shape,out_ndim,out_shape)){
            set_err("multi-output with different shape not supported"); return false;
        }
    }

    bool *reach = calloc(g->nnodes, sizeof(bool));
    if (!reach) { set_err("oom"); return false; }
    mark_reachable(g, reach);

    // map input node id -> position in g->inputs
    int *node_to_inpos = malloc(g->nnodes*sizeof(int));
    if (!node_to_inpos){ free(reach); set_err("oom"); return false; }
    for (size_t i=0;i<g->nnodes;i++) node_to_inpos[i]=-1;
    for (size_t i=0;i<g->ninputs;i++) node_to_inpos[g->inputs[i]] = (int)i;

    // precompute bstrides for each input
    size_t bstrides[16][PG_MAX_NDIM]={0};
    bool input_is_direct[16]={0};
    bool any_bcast=false;
    for (size_t i=0;i<g->ninputs;i++){
        int nid=g->inputs[i];
        const jnode_t *in=&g->nodes[nid];
        if (!reach[nid]) continue; // unused input, ignore
        if (is_direct(in,out_ndim,out_shape)){
            input_is_direct[i]=true;
        }else{
            input_is_direct[i]=false;
            any_bcast=true;
            if (!compute_bcast_strides(in,out_ndim,out_shape,bstrides[i])){ free(reach); free(node_to_inpos); set_err("failed bcast strides"); return false; }
        }
    }

    fprintf(f, "#include <math.h>\n");
    fprintf(f, "#include <stddef.h>\n");
    fprintf(f, "void pg_jit_kernel(float** outs, const float** ins) {\n");
    for (size_t i=0;i<g->ninputs;i++){
        int nid=g->inputs[i];
        if (!reach[nid]) continue;
        fprintf(f, "  const float* in%zu = ins[%zu];\n", i, i);
    }
    for (size_t i=0;i<g->noutputs;i++){
        fprintf(f, "  float* out%zu = outs[%zu];\n", i, i);
    }
    // For bcast we emit nested loops to avoid div/mod per element (expensive idiv)
    // For direct (no bcast) we keep single flat loop for better vectorization
    if (any_bcast){
        // compute out strides for flat index
        size_t out_stride[PG_MAX_NDIM]={0};
        compute_strides(out_ndim, out_shape, out_stride);
        // emit nested loops
        for (size_t d=0; d<out_ndim; d++){
            // indent = 2 + 2*d spaces
            for (size_t s=0;s<2+2*d;s++) fputc(' ', f);
            fprintf(f, "for (size_t c%zu=0; c%zu<%zu; ++c%zu) {\n", d, d, out_shape[d], d);
        }
        // compute flat index i
        {
            for (size_t s=0;s<2+2*out_ndim;s++) fputc(' ', f);
            fprintf(f, "size_t i = ");
            bool first=true;
            for (size_t d=0; d<out_ndim; d++){
                if (!first) fprintf(f, " + ");
                fprintf(f, "c%zu * %zu", d, out_stride[d]);
                first=false;
            }
            if (first) fprintf(f, "0");
            fprintf(f, ";\n");
        }
        for (size_t i=0;i<g->ninputs;i++){
            int nid=g->inputs[i];
            if (!reach[nid]) continue;
            for (size_t s=0;s<2+2*out_ndim;s++) fputc(' ', f);
            if (input_is_direct[i]){
                fprintf(f, "size_t off_in%zu = i;\n", i);
            }else{
                fprintf(f, "size_t off_in%zu = ", i);
                bool first=true;
                for (size_t d=0; d<out_ndim; d++){
                    if (bstrides[i][d]==0) continue;
                    if (!first) fprintf(f, " + ");
                    fprintf(f, "c%zu * %zu", d, bstrides[i][d]);
                    first=false;
                }
                if (first) fprintf(f, "0");
                fprintf(f, ";\n");
            }
        }
        // emit per-node computation in order with deeper indent
        for (size_t nid=0; nid<g->nnodes; ++nid){
            if (!reach[nid]) continue;
            const jnode_t *n=&g->nodes[nid];
            for (size_t s=0;s<2+2*out_ndim;s++) fputc(' ', f);
            if (n->kind==JNODE_INPUT){
                int pos=node_to_inpos[nid];
                assert(pos>=0);
                fprintf(f, "float v%zu = in%zu[off_in%zu];\n", nid, (size_t)pos, (size_t)pos);
            }else if (n->kind==JNODE_CONST){
                fprintf(f, "float v%zu = (float)%.9g;\n", nid, (double)n->const_val);
            }else if (n->kind==JNODE_OP){
                char expr[512];
                if (op_is_binary(n->op)){
                    int a=n->inputs[0], b=n->inputs[1];
                    char sa[32], sb[32];
                    snprintf(sa,sizeof(sa),"v%d",a);
                    snprintf(sb,sizeof(sb),"v%d",b);
                    op_to_expr(n->op, sa, sb, expr, sizeof(expr));
                    fprintf(f, "float v%zu = %s;\n", nid, expr);
                }else{
                    int a=n->inputs[0];
                    char sa[32];
                    snprintf(sa,sizeof(sa),"v%d",a);
                    op_to_expr(n->op, sa, sa, expr, sizeof(expr));
                    fprintf(f, "float v%zu = %s;\n", nid, expr);
                }
            }
        }
        // store outputs
        for (size_t i=0;i<g->noutputs;i++){
            int oid=g->outputs[i];
            for (size_t s=0;s<2+2*out_ndim;s++) fputc(' ', f);
            fprintf(f, "out%zu[i] = v%d;\n", i, oid);
        }
        for (int d=(int)out_ndim-1; d>=0; --d){
            for (size_t s=0;s<2+2*(size_t)d;s++) fputc(' ', f);
            fprintf(f, "}\n");
        }
    } else {
        fprintf(f, "  for (size_t i=0; i<%zu; ++i) {\n", out_numel);
        for (size_t i=0;i<g->ninputs;i++){
            int nid=g->inputs[i];
            if (!reach[nid]) continue;
            fprintf(f, "    size_t off_in%zu = i;\n", i);
        }
        // emit per-node computation in order
        for (size_t nid=0; nid<g->nnodes; ++nid){
            if (!reach[nid]) continue;
            const jnode_t *n=&g->nodes[nid];
            if (n->kind==JNODE_INPUT){
                int pos=node_to_inpos[nid];
                assert(pos>=0);
                fprintf(f, "    float v%zu = in%zu[off_in%zu];\n", nid, (size_t)pos, (size_t)pos);
            }else if (n->kind==JNODE_CONST){
                fprintf(f, "    float v%zu = (float)%.9g;\n", nid, (double)n->const_val);
            }else if (n->kind==JNODE_OP){
                char expr[512];
                if (op_is_binary(n->op)){
                    int a=n->inputs[0], b=n->inputs[1];
                    char sa[32], sb[32];
                    snprintf(sa,sizeof(sa),"v%d",a);
                    snprintf(sb,sizeof(sb),"v%d",b);
                    op_to_expr(n->op, sa, sb, expr, sizeof(expr));
                    fprintf(f, "    float v%zu = %s;\n", nid, expr);
                }else{
                    int a=n->inputs[0];
                    char sa[32];
                    snprintf(sa,sizeof(sa),"v%d",a);
                    op_to_expr(n->op, sa, sa, expr, sizeof(expr));
                    fprintf(f, "    float v%zu = %s;\n", nid, expr);
                }
            }
        }
        // store outputs
        for (size_t i=0;i<g->noutputs;i++){
            int oid=g->outputs[i];
            fprintf(f, "    out%zu[i] = v%d;\n", i, oid);
        }
        fprintf(f, "  }\n");
    }
    fprintf(f, "}\n");

    free(reach);
    free(node_to_inpos);
    return true;
}

/* ---------- compilation ---------- */
static bool file_exists(const char *p){
    struct stat st; return stat(p,&st)==0;
}
static int compile_c_to_so(const char *c_path, const char *so_path){
    const char *comps[]={"cc","gcc","clang", NULL};
    char cmd[2048];
    for (int i=0; comps[i]; i++){
        // check if compiler exists
        char which[256];
        snprintf(which,sizeof(which),"which %s > /dev/null 2>&1", comps[i]);
        if (system(which)!=0) continue;
        snprintf(cmd,sizeof(cmd),"%s -O3 -ffast-math -fPIC -shared -o %s %s -lm 2>&1",
                 comps[i], so_path, c_path);
        int rc = system(cmd);
        if (rc==0 && file_exists(so_path)) return 0;
    }
    // try plain cc anyway
    snprintf(cmd,sizeof(cmd),"cc -O3 -ffast-math -fPIC -shared -o %s %s -lm 2>&1", so_path, c_path);
    int rc = system(cmd);
    if (rc==0 && file_exists(so_path)) return 0;
    return -1;
}

static void make_temp_paths(char *c_path, size_t csz, char *so_path, size_t ssz){
    // use pid + counter + random
    size_t cnt = __sync_fetch_and_add(&g_jit_counter, 1);
    pid_t pid = getpid();
    unsigned r = (unsigned)rand();
    snprintf(c_path,csz,"/tmp/picograd_jit_%d_%zu_%u.c", (int)pid, cnt, r);
    snprintf(so_path,ssz,"/tmp/picograd_jit_%d_%zu_%u.so", (int)pid, cnt, r);
    // ensure not exists, if exists increment
    int tries=0;
    while((file_exists(c_path)||file_exists(so_path)) && tries<10){
        r++; cnt++;
        snprintf(c_path,csz,"/tmp/picograd_jit_%d_%zu_%u.c", (int)pid, cnt, r);
        snprintf(so_path,ssz,"/tmp/picograd_jit_%d_%zu_%u.so", (int)pid, cnt, r);
        tries++;
    }
}

static pg_jit_exe *exe_new(void){
    pg_jit_exe *e = calloc(1,sizeof(*e));
    return e;
}

pg_jit_exe *pg_jit_compile(pg_jit_graph *g){
    if (!g) { set_err("null graph"); return NULL; }
    if (g->noutputs==0) { set_err("graph has no outputs"); return NULL; }
    if (g->ninputs > 16 || g->noutputs > 16) { set_err("too many inputs/outputs (max 16)"); return NULL; }

    uint64_t h = pg_jit_graph_hash(g);

    // check cache
    if (g_cache_enabled){
        for (cache_entry_t *e=g_cache_head; e; e=e->next){
            if (e->hash==h && graph_equal(g, e->graph_copy)){
                // cache hit: create new exe reusing so_path
                pg_jit_exe *exe = exe_new();
                if (!exe) { set_err("oom"); return NULL; }
                exe->hash = h;
                strncpy(exe->c_path, e->c_path, sizeof(exe->c_path)-1);
                strncpy(exe->so_path, e->so_path, sizeof(exe->so_path)-1);
                exe->ninputs = e->ninputs;
                exe->noutputs = e->noutputs;
                for (size_t i=0;i<e->ninputs;i++){ exe->in_ndim[i]=e->in_ndim[i]; memcpy(exe->in_shape[i], e->in_shape[i], e->in_ndim[i]*sizeof(size_t)); }
                for (size_t i=0;i<e->noutputs;i++){ exe->out_ndim[i]=e->out_ndim[i]; memcpy(exe->out_shape[i], e->out_shape[i], e->out_ndim[i]*sizeof(size_t)); exe->out_numel[i]=e->out_numel[i]; }
                // dlopen
                void *hdl = dlopen(exe->so_path, RTLD_NOW);
                if (!hdl){ set_err("dlopen cache hit failed: %s", dlerror()); free(exe); return NULL; }
                void (*kern)(float**,const float**) = (void(*)(float**,const float**))dlsym(hdl, "pg_jit_kernel");
                if (!kern){ set_err("dlsym failed: %s", dlerror()); dlclose(hdl); free(exe); return NULL; }
                exe->handle = hdl;
                exe->kernel = kern;
                return exe;
            }
        }
    }

    // generate C code to temp file
    char c_path[512], so_path[512];
    make_temp_paths(c_path,sizeof(c_path),so_path,sizeof(so_path));

    FILE *f = fopen(c_path,"w");
    if (!f){ set_err("fopen %s: %s", c_path, strerror(errno)); return NULL; }
    bool ok = emit_c_code(g,f);
    fclose(f);
    if (!ok){
        unlink(c_path);
        return NULL;
    }

    // compile
    if (compile_c_to_so(c_path, so_path)!=0){
        set_err("compile failed for %s (see %s)", so_path, c_path);
        return NULL;
    }

    // dlopen
    void *hdl = dlopen(so_path, RTLD_NOW);
    if (!hdl){
        set_err("dlopen %s failed: %s", so_path, dlerror());
        unlink(c_path); unlink(so_path);
        return NULL;
    }
    void (*kern)(float**,const float**) = (void(*)(float**,const float**))dlsym(hdl, "pg_jit_kernel");
    if (!kern){
        set_err("dlsym pg_jit_kernel failed: %s", dlerror());
        dlclose(hdl);
        unlink(c_path); unlink(so_path);
        return NULL;
    }

    pg_jit_exe *exe = exe_new();
    if (!exe){ dlclose(hdl); unlink(c_path); unlink(so_path); set_err("oom"); return NULL; }
    exe->handle = hdl;
    exe->kernel = kern;
    strncpy(exe->c_path, c_path, sizeof(exe->c_path)-1);
    strncpy(exe->so_path, so_path, sizeof(exe->so_path)-1);
    exe->hash = h;
    exe->ninputs = g->ninputs;
    exe->noutputs = g->noutputs;
    for (size_t i=0;i<g->ninputs;i++){
        int nid=g->inputs[i];
        const jnode_t *n=&g->nodes[nid];
        exe->in_ndim[i]=n->ndim;
        memcpy(exe->in_shape[i], n->shape, n->ndim*sizeof(size_t));
    }
    for (size_t i=0;i<g->noutputs;i++){
        int oid=g->outputs[i];
        const jnode_t *n=&g->nodes[oid];
        exe->out_ndim[i]=n->ndim;
        memcpy(exe->out_shape[i], n->shape, n->ndim*sizeof(size_t));
        exe->out_numel[i]=n->numel;
    }

    // insert into cache
    if (g_cache_enabled){
        cache_entry_t *ce = calloc(1,sizeof(*ce));
        if (ce){
            ce->hash = h;
            strncpy(ce->c_path, c_path, sizeof(ce->c_path)-1);
            strncpy(ce->so_path, so_path, sizeof(ce->so_path)-1);
            ce->ninputs = exe->ninputs;
            ce->noutputs = exe->noutputs;
            for (size_t i=0;i<exe->ninputs;i++){ ce->in_ndim[i]=exe->in_ndim[i]; memcpy(ce->in_shape[i], exe->in_shape[i], exe->in_ndim[i]*sizeof(size_t)); }
            for (size_t i=0;i<exe->noutputs;i++){ ce->out_ndim[i]=exe->out_ndim[i]; memcpy(ce->out_shape[i], exe->out_shape[i], exe->out_ndim[i]*sizeof(size_t)); ce->out_numel[i]=exe->out_numel[i]; }
            ce->graph_copy = graph_clone(g);
            ce->next = g_cache_head;
            g_cache_head = ce;
            g_cache_size++;
        }
    }

    return exe;
}

void pg_jit_exe_free(pg_jit_exe *exe){
    if (!exe) return;
    if (exe->handle) dlclose(exe->handle);
    // do not unlink files if they are in cache (cache owns them)
    bool in_cache=false;
    if (g_cache_enabled){
        for (cache_entry_t *e=g_cache_head; e; e=e->next){
            if (strcmp(e->so_path, exe->so_path)==0) { in_cache=true; break; }
        }
    }
    if (!in_cache){
        if (exe->c_path[0]) unlink(exe->c_path);
        if (exe->so_path[0]) unlink(exe->so_path);
    }
    free(exe);
}

bool pg_jit_run(pg_jit_exe *exe, const pg_tensor **inputs, size_t ninputs,
                pg_tensor **outputs, size_t noutputs){
    if (!exe || !inputs || !outputs) { set_err("null args"); return false; }
    if (ninputs != exe->ninputs || noutputs != exe->noutputs){ set_err("mismatched ninputs/noutputs: got %zu/%zu expected %zu/%zu", ninputs,noutputs,exe->ninputs,exe->noutputs); return false; }
    for (size_t i=0;i<ninputs;i++){
        if (!inputs[i] || !inputs[i]->data){ set_err("null input %zu",i); return false; }
        if (inputs[i]->ndim != exe->in_ndim[i] || !shape_equal(inputs[i]->ndim, inputs[i]->shape, exe->in_ndim[i], exe->in_shape[i])){
            set_err("input %zu shape mismatch",i); return false;
        }
    }
    for (size_t i=0;i<noutputs;i++){
        if (!outputs[i] || !outputs[i]->data){ set_err("null output %zu",i); return false; }
        if (outputs[i]->ndim != exe->out_ndim[i] || !shape_equal(outputs[i]->ndim, outputs[i]->shape, exe->out_ndim[i], exe->out_shape[i])){
            set_err("output %zu shape mismatch",i); return false;
        }
    }
    float *out_ptrs[16];
    const float *in_ptrs[16];
    for (size_t i=0;i<ninputs;i++) in_ptrs[i]=inputs[i]->data;
    for (size_t i=0;i<noutputs;i++) out_ptrs[i]=outputs[i]->data;
    // call kernel; kernel expects float** and const float**
    exe->kernel(out_ptrs, (const float**)in_ptrs);
    return true;
}

pg_tensor *pg_jit_run_single(pg_jit_exe *exe, const pg_tensor **inputs, size_t ninputs){
    if (!exe) { set_err("null exe"); return NULL; }
    if (exe->noutputs!=1){ set_err("run_single expects 1 output, got %zu", exe->noutputs); return NULL; }
    pg_tensor *out = pg_tensor_new(exe->out_ndim[0], exe->out_shape[0]);
    if (!out){ set_err("alloc output failed"); return NULL; }
    pg_tensor *outs[1]={out};
    if (!pg_jit_run(exe, inputs, ninputs, outs, 1)){ pg_tensor_free(out); return NULL; }
    return out;
}

void pg_jit_cache_clear(void){
    cache_entry_t *e=g_cache_head;
    while(e){
        cache_entry_t *n=e->next;
        if (e->so_path[0]) unlink(e->so_path);
        if (e->c_path[0]) unlink(e->c_path);
        pg_jit_graph_free(e->graph_copy);
        free(e);
        e=n;
    }
    g_cache_head=NULL;
    g_cache_size=0;
}
