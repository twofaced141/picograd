#include "../backend_i.h"
#include "driver.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BTILE 64
#define TPT 16
#define PG_HIP_THREADS 256

/* hipMemcpyKind values - must match hip_runtime_api.h */
#define HIP_MEMCPY_H2D 1
#define HIP_MEMCPY_D2H 2
#define HIP_MEMCPY_D2D 3

static void *g_module = NULL;
static void *g_fn_sgemm = NULL;
static void *g_fn_map = NULL;
static void *g_fn_bin = NULL;
static void *g_fn_accum_scatter = NULL;
static void *g_fn_sum_axis = NULL;
static void *g_fn_softmax = NULL;
static void *g_fn_copy_strided = NULL;
static void *g_fn_fill = NULL;

/* ---------- embedded HIP kernel source (compiled via hiprtc at runtime) ---------- */

static const char *hip_kernel_source(void)
{
    return
    "#define PG_MAX_OP_NDIM 8\n"
    "#define PG_MAP_EXP 0\n"
    "#define PG_MAP_LOG 1\n"
    "#define PG_MAP_SIN 2\n"
    "#define PG_MAP_COS 3\n"
    "#define PG_MAP_SQRT 4\n"
    "#define PG_MAP_NEG 5\n"
    "#define PG_MAP_ABS 6\n"
    "#define PG_MAP_ERF 7\n"
    "#define PG_MAP_RELU 8\n"
    "#define PG_MAP_SIGMOID 9\n"
    "#define PG_MAP_TANH 10\n"
    "#define PG_BIN_ADD 0\n"
    "#define PG_BIN_SUB 1\n"
    "#define PG_BIN_MUL 2\n"
    "#define PG_BIN_DIV 3\n"
    "#define PG_BIN_SIG_BW 4\n"
    "#define PG_BIN_TANH_BW 5\n"
    "#define PG_BIN_RELU_BW 6\n"
    "\n"
    "typedef unsigned int u32;\n"
    "struct PgBinArgs { u32 ndim; u32 numel; u32 shape[8]; u32 sa[8]; u32 sb[8]; };\n"
    "struct PgStrides { u32 ndim; u32 numel; u32 shape[8]; u32 s[8]; };\n"
    "\n"
    "__device__ float pg_expf(float x) {\n"
    "    if (x > 88.0f) return 3.4028235e38f;\n"
    "    if (x < -88.0f) return 0.0f;\n"
    "    const float LOG2E = 1.442695041f;\n"
    "    const float LN2 = 0.693147181f;\n"
    "    float y = x * LOG2E;\n"
    "    int k = (int)(y >= 0.0f ? y + 0.5f : y - 0.5f);\n"
    "    float r = x - (float)k * LN2;\n"
    "    float p = 1.0f + r*(1.0f + r*(0.5f + r*(0.166666667f + r*(0.041666667f + r*(0.008333333f + r*0.001388889f)))));\n"
    "    union { unsigned u; float f; } sc; sc.u = (unsigned)((k+127)<<23);\n"
    "    return p * sc.f;\n"
    "}\n"
    "__device__ float pg_logf(float x) {\n"
    "    union { unsigned u; float f; } v; v.f = x;\n"
    "    int e = (int)(v.u>>23)-127; v.u = (v.u & 0x007fffffu) | 0x3f800000u; float m=v.f;\n"
    "    if(m>1.414213562f){m*=0.5f;e++;}\n"
    "    float s=(m-1.0f)/(m+1.0f); float z=s*s;\n"
    "    float p=s*(2.0f+z*(0.666666667f+z*(0.4f+z*(0.285714286f+z*(0.222222222f+z*0.181818182f)))));\n"
    "    return p + (float)e*0.693147181f;\n"
    "}\n"
    "__device__ float pg_sinf(float x) {\n"
    "    const float TWO_OVER_PI=0.636619772f; const float PI_2A=1.5707855225f; const float PI_2B=7.1386587e-7f;\n"
    "    float q=x*TWO_OVER_PI; int quad=(int)(q>=0.0f?q+0.5f:q-0.5f);\n"
    "    float r=x-(float)quad*PI_2A-(float)quad*PI_2B; float r2=r*r;\n"
    "    float sp=r + r*r2*(-0.166666567f + r2*(0.008333025f - r2*0.000198074f));\n"
    "    float cp=1.0f + r2*(-0.499999925f + r2*(0.041645982f - r2*0.001358558f));\n"
    "    switch(quad & 3){ case 0: return sp; case 1: return cp; case 2: return -sp; default: return -cp; }\n"
    "}\n"
    "__device__ float pg_cosf(float x){ return pg_sinf(x+1.570796327f); }\n"
    "__device__ float pg_tanhf(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; float e2=pg_expf(2.0f*x); return (e2-1.0f)/(e2+1.0f); }\n"
    "__device__ float pg_erff(float x){ float ax=x<0.0f?-x:x; float t=1.0f/(1.0f+0.3275911f*ax); float p=t*(0.254829592f + t*(-0.284496736f + t*(1.421413741f + t*(-1.453152027f + t*1.061405429f)))) ; float r=1.0f - p*pg_expf(-x*x); return x<0.0f?-r:r; }\n"
    "__device__ float pg_sqrtf(float x){ if(x<=0.0f) return 0.0f; union{unsigned u; float f;} v; v.f=x; v.u=(v.u>>1)+0x1fc00000u; float g=v.f; for(int it=0;it<4;it++) g=0.5f*(g+x/g); return g; }\n"
    "__device__ float map_apply(int op,float x){ switch(op){ case PG_MAP_EXP: return pg_expf(x); case PG_MAP_LOG: return pg_logf(x); case PG_MAP_SIN: return pg_sinf(x); case PG_MAP_COS: return pg_cosf(x); case PG_MAP_SQRT: return pg_sqrtf(x); case PG_MAP_NEG: return -x; case PG_MAP_ABS: return x<0.0f?-x:x; case PG_MAP_ERF: return pg_erff(x); case PG_MAP_RELU: return x>0.0f?x:0.0f; case PG_MAP_SIGMOID: return 1.0f/(1.0f+pg_expf(-x)); case PG_MAP_TANH: return pg_tanhf(x); default: return x; } }\n"
    "__device__ float bin_apply(int op,float a,float b){ switch(op){ case PG_BIN_ADD: return a+b; case PG_BIN_SUB: return a-b; case PG_BIN_MUL: return a*b; case PG_BIN_DIV: return a/b; case PG_BIN_SIG_BW: return b*a*(1.0f-a); case PG_BIN_TANH_BW: return b*(1.0f-a*a); case PG_BIN_RELU_BW: return a>0.0f?b:0.0f; default: return a; } }\n"
    "\n"
    "extern \"C\" __global__ void pg_k_map(float* out, const float* src, unsigned int n, unsigned int op){\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<n;i+=stride) out[i]=map_apply((int)op,src[i]);\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_bin(float* out, const float* a, const float* b, unsigned int op, struct PgBinArgs ar){\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<ar.numel;i+=stride){\n"
    "        unsigned int ind[8]; unsigned int rem=i;\n"
    "        for(unsigned int d=ar.ndim; d-- >0;){ ind[d]=rem % ar.shape[d]; rem/=ar.shape[d]; }\n"
    "        unsigned int oa=0,ob=0; for(unsigned int d=0;d<ar.ndim;d++){ oa+=ind[d]*ar.sa[d]; ob+=ind[d]*ar.sb[d]; }\n"
    "    out[i]=bin_apply((int)op,a[oa],b[ob]);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_accum_scatter(float* dst, const float* src, float scale, struct PgStrides ar){\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<ar.numel;i+=stride){\n"
    "        unsigned int ind[8]; unsigned int rem=i;\n"
    "        for(unsigned int d=ar.ndim; d-- >0;){ ind[d]=rem % ar.shape[d]; rem/=ar.shape[d]; }\n"
    "        unsigned int od=0; for(unsigned int d=0;d<ar.ndim;d++) od+=ind[d]*ar.s[d];\n"
    "        dst[od] += scale * src[i];\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_sum_axis(float* out, const float* src, float scale, unsigned int outer, unsigned int len, unsigned int inner, unsigned int ostride){\n"
    "    unsigned int total = outer*inner;\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<total;i+=stride){\n"
    "        unsigned int o=i/inner; unsigned int ii=i%inner; float acc=0.0f;\n"
    "        const float* p = src + (size_t)o*len*inner + ii;\n"
    "        for(unsigned int j=0;j<len;j++,p+=inner) acc+=*p;\n"
    "        out[(size_t)o*ostride + ii] = acc * scale;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_softmax(float* out, const float* src, unsigned int outer, unsigned int len, unsigned int inner){\n"
    "    unsigned int total=outer*inner;\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<total;i+=stride){\n"
    "        unsigned int o=i/inner; unsigned int ii=i%inner;\n"
    "        const float* col=src + (size_t)o*len*inner + ii;\n"
    "        float* dcol=out + (size_t)o*len*inner + ii;\n"
    "        float mx=col[0]; for(unsigned int j=1;j<len;j++){ float v=col[(size_t)j*inner]; if(v>mx) mx=v; }\n"
    "        float sum=0.0f; for(unsigned int j=0;j<len;j++){ float e=pg_expf(col[(size_t)j*inner]-mx); dcol[(size_t)j*inner]=e; sum+=e; }\n"
    "        float inv=1.0f/sum; for(unsigned int j=0;j<len;j++) dcol[(size_t)j*inner]*=inv;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_copy_strided(float* dst, const float* src, struct PgStrides ar){\n"
    "    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned int stride = blockDim.x * gridDim.x;\n"
    "    for(unsigned int i=idx;i<ar.numel;i+=stride){\n"
    "        unsigned int ind[8]; unsigned int rem=i;\n"
    "        for(unsigned int d=ar.ndim; d-- >0;){ ind[d]=rem % ar.shape[d]; rem/=ar.shape[d]; }\n"
    "        unsigned int os=0; for(unsigned int d=0;d<ar.ndim;d++) os+=ind[d]*ar.s[d];\n"
    "        dst[i]=src[os];\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void pg_k_fill(float* p, unsigned int n, float v){ unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x; unsigned int stride=blockDim.x*gridDim.x; for(unsigned int i=idx;i<n;i+=stride) p[i]=v; }\n"
    "\n"
    "#define BM 64\n"
    "#define BN 64\n"
    "#define BK 32\n"
    "#define RT 4\n"
    "#define RN 4\n"
    "#define TP 16\n"
    "extern \"C\" __global__ void pg_sgemm_kernel(const float* a, const float* b, float* c, unsigned int m, unsigned int n, unsigned int k){\n"
    "    unsigned int tx = threadIdx.x; unsigned int ty = threadIdx.y;\n"
    "    unsigned int tid = ty * TP + tx;\n"
    "    unsigned int row0 = blockIdx.y * BM + ty * RT;\n"
    "    unsigned int col0 = blockIdx.x * BN + tx * RN;\n"
    "    __shared__ float s_a[BM*BK]; __shared__ float s_b[BK*BN];\n"
    "    float acc[RT][RN]; for(int i=0;i<RT;i++) for(int j=0;j<RN;j++) acc[i][j]=0.0f;\n"
    "    unsigned int ntiles = (k + BK - 1)/BK;\n"
    "    for(unsigned int t=0; t<ntiles; t++){\n"
    "        unsigned int gk = t * BK;\n"
    "        for(unsigned int idx=tid; idx<BM*BK; idx+=TP*TP){\n"
    "            unsigned int r=idx / BK; unsigned int cc=idx % BK;\n"
    "            unsigned int gr = blockIdx.y * BM + r; unsigned int gc = gk + cc;\n"
    "            s_a[idx] = (gr < m && gc < k) ? a[(size_t)gr * k + gc] : 0.0f;\n"
    "        }\n"
    "        for(unsigned int idx=tid; idx<BK*BN; idx+=TP*TP){\n"
    "            unsigned int rr=idx / BN; unsigned int cc=idx % BN;\n"
    "            unsigned int gr = gk + rr; unsigned int gc = blockIdx.x * BN + cc;\n"
    "            s_b[idx] = (gr < k && gc < n) ? b[(size_t)gr * n + gc] : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        for(unsigned int kk=0; kk<BK; kk++){\n"
    "            float ra[RT], rb[RN];\n"
    "            for(int i=0;i<RT;i++) ra[i]=s_a[(ty*RT + i)*BK + kk];\n"
    "            for(int j=0;j<RN;j++) rb[j]=s_b[kk*BN + tx*RN + j];\n"
    "            for(int i=0;i<RT;i++) for(int j=0;j<RN;j++) acc[i][j]+=ra[i]*rb[j];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    for(int i=0;i<RT;i++){ unsigned int gr=row0+i; if(gr>=m) continue; for(int j=0;j<RN;j++){ unsigned int gc=col0+j; if(gc<n) c[(size_t)gr * n + gc]=acc[i][j]; }\n"
    "    }\n"
    "}\n"
    ;
}

/* ---------- helpers for hiprtc compilation ---------- */

static pg_status hip_compile_and_load(const char *src, void **out_module)
{
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv || !drv->rtcCreateProgram)
        return PG_ERR_UNSUPPORTED;

    void *prog = NULL;
    int rc = drv->rtcCreateProgram(&prog, src, "picograd_hip", 0, NULL, NULL);
    if (rc != 0 || !prog)
        return PG_ERR_GEMM;

    /* Try compile with minimal options first, then fallback with arch-specific */
    const char *opts0[] = { "-O3", "-std=c++14" };
    const char *opts1[] = { "-O3", "-std=c++14", "--gpu-architecture=gfx900" };
    const char *opts2[] = { "-O3", "-std=c++14", "--gpu-architecture=gfx906" };
    const char *opts3[] = { "-O3", "-std=c++14", "--gpu-architecture=gfx1030" };
    const char *opts4[] = { "-O3", "-std=c++14", "--gpu-architecture=gfx1100" };

    const struct { int n; const char **opts; } attempts[] = {
        { 2, opts0 },
        { 3, opts1 },
        { 3, opts2 },
        { 3, opts3 },
        { 3, opts4 },
    };

    int compiled = 0;
    for (size_t a = 0; a < sizeof(attempts)/sizeof(attempts[0]); a++) {
        rc = drv->rtcCompileProgram(prog, attempts[a].n, attempts[a].opts);
        if (rc == 0) { compiled = 1; break; }
        /* if failed, try next - but destroy and recreate program for some hiprtc versions */
        if (getenv("PG_HIP_DEBUG")) {
            size_t logSize = 0;
            drv->rtcGetProgramLogSize(prog, &logSize);
            if (logSize > 1) {
                char *log = (char*)malloc(logSize);
                if (log) {
                    drv->rtcGetProgramLog(prog, log);
                    fprintf(stderr, "[hip] hiprtc attempt %zu failed: %s\n", a, log);
                    free(log);
                }
            }
        }
        if (a + 1 < sizeof(attempts)/sizeof(attempts[0])) {
            /* recreate program for next attempt */
            drv->rtcDestroyProgram(&prog);
            prog = NULL;
            if (drv->rtcCreateProgram(&prog, src, "picograd_hip", 0, NULL, NULL) != 0 || !prog)
                break;
        }
    }

    if (!compiled) {
        if (getenv("PG_HIP_DEBUG")) {
            size_t logSize = 0;
            drv->rtcGetProgramLogSize(prog, &logSize);
            char *log = (char*)malloc(logSize ? logSize : 1);
            if (log) {
                drv->rtcGetProgramLog(prog, log);
                fprintf(stderr, "[hip] hiprtc compile final failure: %s\n", log);
                free(log);
            }
        }
        drv->rtcDestroyProgram(&prog);
        return PG_ERR_GEMM;
    }

    size_t codeSize = 0;
    rc = drv->rtcGetCodeSize(prog, &codeSize);
    if (rc != 0 || codeSize == 0) {
        drv->rtcDestroyProgram(&prog);
        return PG_ERR_GEMM;
    }
    char *code = (char*)malloc(codeSize);
    if (!code) {
        drv->rtcDestroyProgram(&prog);
        return PG_ERR_ALLOC;
    }
    rc = drv->rtcGetCode(prog, code);
    drv->rtcDestroyProgram(&prog);
    if (rc != 0) {
        free(code);
        return PG_ERR_GEMM;
    }

    void *mod = NULL;
    rc = drv->moduleLoadData(&mod, code);
    free(code);
    if (rc != 0 || !mod)
        return PG_ERR_GEMM;

    *out_module = mod;
    return PG_OK;
}

static void hip_register_gpu(void);

static pg_status hip_init(void)
{
    static pg_status cached = PG_ERR_UNSUPPORTED;
    static int done = 0;
    if (done) return cached;
    done = 1;

    int debug = getenv("PG_HIP_DEBUG") != NULL;
    pg_status err = PG_OK;
    const pg_hip_drv *drv = pg_hip_drv_get(&err);
    if (!drv) {
        if (debug) fprintf(stderr, "picograd/hip: driver get failed %d\n", err);
        return cached = err;
    }

    int count = 0;
    if (drv->getDeviceCount(&count) != 0 || count == 0) {
        if (debug) fprintf(stderr, "picograd/hip: no devices count=%d\n", count);
        return cached = PG_ERR_UNSUPPORTED;
    }
    if (drv->setDevice(0) != 0) {
        if (debug) fprintf(stderr, "picograd/hip: hipSetDevice 0 failed\n");
        return cached = PG_ERR_UNSUPPORTED;
    }

    /* Try to compile kernels via hiprtc */
    const char *src = hip_kernel_source();
    void *mod = NULL;
    pg_status st = hip_compile_and_load(src, &mod);
    if (st != PG_OK) {
        if (debug) fprintf(stderr, "picograd/hip: kernel JIT failed %d\n", st);
        return cached = st;
    }

    g_module = mod;

    int ok = 1;
    int rc;
    rc = drv->moduleGetFunction(&g_fn_sgemm, g_module, "pg_sgemm_kernel");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_sgemm_kernel -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_map, g_module, "pg_k_map");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_map -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_bin, g_module, "pg_k_bin");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_bin -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_accum_scatter, g_module, "pg_k_accum_scatter");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_accum_scatter -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_sum_axis, g_module, "pg_k_sum_axis");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_sum_axis -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_softmax, g_module, "pg_k_softmax");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_softmax -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_copy_strided, g_module, "pg_k_copy_strided");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_copy_strided -> %d\n", rc); ok = 0; }
    rc = drv->moduleGetFunction(&g_fn_fill, g_module, "pg_k_fill");
    if (rc != 0) { if (debug) fprintf(stderr, "picograd/hip: get pg_k_fill -> %d (non-fatal)\n", rc); /* fill optional */ }

    if (ok) {
        hip_register_gpu();
        if (debug) fprintf(stderr, "picograd/hip: init ok (sgemm+ops)\n");
    } else {
        if (debug) fprintf(stderr, "picograd/hip: init ok (partial, some kernels missing, fallback to CPU for those)\n");
        /* still register whatever succeeded? we already have part; register only available */
        hip_register_gpu();
    }

    cached = PG_OK;
    return cached;
}

/* ---------- HIP runtime wrappers ---------- */

static void *hip_malloc(size_t nbytes)
{
    if (hip_init() != PG_OK) return NULL;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return NULL;
    void *ptr = NULL;
    size_t sz = nbytes ? nbytes : 1;
    if (drv->malloc(&ptr, sz) != 0) return NULL;
    return ptr;
}

static void hip_free(void *p)
{
    if (!p) return;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return;
    drv->free(p);
}

static pg_status hip_h2d(void *dst, const void *src, size_t nbytes)
{
    if (hip_init() != PG_OK) return PG_ERR_COPY;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_COPY;
    return drv->memcpy(dst, src, nbytes, HIP_MEMCPY_H2D) == 0 ? PG_OK : PG_ERR_COPY;
}

static pg_status hip_d2h(void *dst, const void *src, size_t nbytes)
{
    if (hip_init() != PG_OK) return PG_ERR_COPY;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_COPY;
    return drv->memcpy(dst, src, nbytes, HIP_MEMCPY_D2H) == 0 ? PG_OK : PG_ERR_COPY;
}

static pg_status hip_sync(void)
{
    if (hip_init() != PG_OK) return PG_ERR_SYNC;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_SYNC;
    return drv->deviceSynchronize() == 0 ? PG_OK : PG_ERR_SYNC;
}

static void hip_gemm(size_t m, size_t n, size_t k,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     float *c, size_t ldc)
{
    assert(lda == k && ldb == n && ldc == n);
    assert(m <= 0xffffffffu && n <= 0xffffffffu && k <= 0xffffffffu);
    if (hip_init() != PG_OK || !g_fn_sgemm) {
        /* unreachable: caller ensures HIP backend is active */
        assert(!"hip backend gemm not available");
        return;
    }
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    assert(drv);
    unsigned m32 = (unsigned)m, n32 = (unsigned)n, k32 = (unsigned)k;
    void *params[] = { (void*)&a, (void*)&b, (void*)&c, &m32, &n32, &k32 };
    unsigned gx = (n32 + BTILE - 1) / BTILE;
    unsigned gy = (m32 + BTILE - 1) / BTILE;
    int rc = drv->moduleLaunchKernel(g_fn_sgemm, gx, gy, 1, TPT, TPT, 1, 0, NULL, params, NULL);
    assert(rc == 0);
    (void)rc;
}

static pg_status hip_gpu_map(float *out, const float *src, size_t n, int op)
{
    if (hip_init() != PG_OK || !g_fn_map) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    unsigned n32 = (unsigned)n;
    unsigned op32 = (unsigned)op;
    void *params[] = { &out, &src, &n32, &op32 };
    unsigned gx = ((unsigned)n + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_map, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_bin(float *out, const float *a, const float *b,
                             size_t n, int op, const pg_k_bin_args *args)
{
    if (hip_init() != PG_OK || !g_fn_bin) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    unsigned op32 = (unsigned)op;
    /* need a local copy to take address for driver */
    pg_k_bin_args kargs = *args;
    void *params[] = { &out, &a, &b, &op32, &kargs };
    unsigned gx = ((unsigned)n + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_bin, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_accum_scatter(float *dst, const float *src, float scale, const pg_k_strides *args)
{
    if (hip_init() != PG_OK || !g_fn_accum_scatter) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    pg_k_strides kargs = *args;
    void *params[] = { &dst, &src, &scale, &kargs };
    unsigned gx = ((unsigned)args->numel + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_accum_scatter, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_sum_axis(float *out, const float *src, float scale,
                                  size_t outer, size_t len, size_t inner,
                                  size_t keepdim_stride)
{
    if (hip_init() != PG_OK || !g_fn_sum_axis) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    unsigned o32 = (unsigned)outer, l32 = (unsigned)len, i32 = (unsigned)inner;
    unsigned ks32 = (unsigned)keepdim_stride;
    void *params[] = { &out, &src, &scale, &o32, &l32, &i32, &ks32 };
    unsigned gx = ((unsigned)(outer * inner) + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_sum_axis, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_softmax(float *out, const float *src,
                                 size_t outer, size_t len, size_t inner)
{
    if (hip_init() != PG_OK || !g_fn_softmax) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    unsigned o32 = (unsigned)outer, l32 = (unsigned)len, i32 = (unsigned)inner;
    void *params[] = { &out, &src, &o32, &l32, &i32 };
    unsigned gx = ((unsigned)(outer * inner) + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_softmax, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_copy_strided(float *dst, const float *src, const pg_k_strides *args)
{
    if (hip_init() != PG_OK || !g_fn_copy_strided) return PG_ERR_GEMM;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_GEMM;
    pg_k_strides kargs = *args;
    void *params[] = { &dst, &src, &kargs };
    unsigned gx = ((unsigned)args->numel + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
    int rc = drv->moduleLaunchKernel(g_fn_copy_strided, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
    return rc == 0 ? PG_OK : PG_ERR_GEMM;
}

static pg_status hip_gpu_fill(void *p, size_t nbytes, float v)
{
    if (hip_init() != PG_OK) return PG_ERR_COPY;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_COPY;

    if (g_fn_fill) {
        size_t n = nbytes / sizeof(float);
        if (n > 0xffffffffu) return PG_ERR_COPY;
        unsigned n32 = (unsigned)n;
        float *fp = (float*)p;
        void *params[] = { &fp, &n32, &v };
        unsigned gx = (n32 + PG_HIP_THREADS - 1) / PG_HIP_THREADS;
        if (gx == 0) gx = 1;
        int rc = drv->moduleLaunchKernel(g_fn_fill, gx, 1, 1, PG_HIP_THREADS, 1, 1, 0, NULL, params, NULL);
        if (rc == 0) return PG_OK;
        /* fall through to memcpy path on launch failure */
    }

    /* fallback for byte-repeated values via memset, otherwise host fill */
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    uint8_t b0 = bits & 0xff;
    uint32_t rep = b0 | (b0<<8) | (b0<<16) | (b0<<24);
    if (rep == bits) {
        /* hipMemset writes byte value */
        if (drv->memset(p, (int)b0, nbytes) == 0) {
            return PG_OK;
        }
    }
    /* host fill + H2D copy */
    float *tmp = (float*)malloc(nbytes);
    if (!tmp) return PG_ERR_ALLOC;
    size_t n = nbytes / sizeof(float);
    for (size_t i = 0; i < n; i++) tmp[i] = v;
    pg_status st = drv->memcpy(p, tmp, nbytes, HIP_MEMCPY_H2D) == 0 ? PG_OK : PG_ERR_COPY;
    free(tmp);
    return st;
}

static pg_status hip_gpu_copy_d2d(void *dst, const void *src, size_t nbytes)
{
    if (hip_init() != PG_OK) return PG_ERR_COPY;
    const pg_hip_drv *drv = pg_hip_drv_get(NULL);
    if (!drv) return PG_ERR_COPY;
    return drv->memcpy(dst, src, nbytes, HIP_MEMCPY_D2D) == 0 ? PG_OK : PG_ERR_COPY;
}

static void hip_register_gpu(void)
{
    pg_gpu.map           = hip_gpu_map;
    pg_gpu.bin           = hip_gpu_bin;
    pg_gpu.accum_scatter = hip_gpu_accum_scatter;
    pg_gpu.sum_axis      = hip_gpu_sum_axis;
    pg_gpu.softmax       = hip_gpu_softmax;
    pg_gpu.copy_strided  = hip_gpu_copy_strided;
    pg_gpu.fill          = hip_gpu_fill;
    pg_gpu.copy_d2d      = hip_gpu_copy_d2d;
}

const pg_backend_ops pg_backend_hip = {
    .name = "hip",
    .init = hip_init,
    .malloc = hip_malloc,
    .free = hip_free,
    .copy_h2d = hip_h2d,
    .copy_d2h = hip_d2h,
    .sync = hip_sync,
    .gemm = hip_gemm,
};
