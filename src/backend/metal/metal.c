#include "../backend_i.h"

#if defined(PICOGRAD_BACKEND_METAL)

#if defined(__APPLE__)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PG_METAL_THREADS 256

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_p_sgemm;
static id<MTLComputePipelineState> g_p_map;
static id<MTLComputePipelineState> g_p_bin;
static id<MTLComputePipelineState> g_p_accum_gather;
static id<MTLComputePipelineState> g_p_accum_scatter;
static id<MTLComputePipelineState> g_p_sum_axis;
static id<MTLComputePipelineState> g_p_softmax;
static id<MTLComputePipelineState> g_p_copy_strided;

/* ---------- embedded MSL kernel source ---------- */

static const char *msl_source(void)
{
    return
    "typedef unsigned int u32;\n"
    "\n"
    "struct BinArgs {\n"
    "    u32 ndim; u32 numel;\n"
    "    u32 shape[8]; u32 sa[8]; u32 sb[8];\n"
    "};\n"
    "\n"
    "struct Strides {\n"
    "    u32 ndim; u32 numel;\n"
    "    u32 shape[8]; u32 s[8];\n"
    "};\n"
    "\n"
    "/* ---- unary map ---- */\n"
    "kernel void pg_k_map(\n"
    "    device float *out [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant u32 &n [[buffer(2)]],\n"
    "    constant u32 &op [[buffer(3)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (tid >= n) return;\n"
    "    float x = src[tid];\n"
    "    float r;\n"
    "    switch (op) {\n"
    "    case 0:  r = exp(x); break;\n"
    "    case 1:  r = log(x); break;\n"
    "    case 2:  r = sin(x); break;\n"
    "    case 3:  r = cos(x); break;\n"
    "    case 4:  r = sqrt(x); break;\n"
    "    case 5:  r = -x; break;\n"
    "    case 6:  r = abs(x); break;\n"
    "    case 7:  r = erf(x); break;\n"
    "    case 8:  r = max(x, 0.0); break;\n"
    "    case 9:  r = 1.0 / (1.0 + exp(-x)); break;\n"
    "    case 10: r = tanh(x); break;\n"
    "    default: r = x; break;\n"
    "    }\n"
    "    out[tid] = r;\n"
    "}\n"
    "\n"
    "/* ---- binary op with broadcast ---- */\n"
    "kernel void pg_k_bin(\n"
    "    device float *out [[buffer(0)]],\n"
    "    device const float *a [[buffer(1)]],\n"
    "    device const float *b [[buffer(2)]],\n"
    "    constant u32 &op [[buffer(3)]],\n"
    "    constant BinArgs &ar [[buffer(4)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (tid >= ar.numel) return;\n"
    "    u32 idx[8];\n"
    "    u32 rem = tid;\n"
    "    for (u32 d = ar.ndim; d-- > 0;) {\n"
    "        idx[d] = rem % ar.shape[d];\n"
    "        rem /= ar.shape[d];\n"
    "    }\n"
    "    u32 oa = 0, ob = 0;\n"
    "    for (u32 d = 0; d < ar.ndim; d++) {\n"
    "        oa += idx[d] * ar.sa[d];\n"
    "        ob += idx[d] * ar.sb[d];\n"
    "    }\n"
    "    float va = a[oa], vb = b[ob];\n"
    "    float r;\n"
    "    switch (op) {\n"
    "    case 0: r = va + vb; break;\n"
    "    case 1: r = va - vb; break;\n"
    "    case 2: r = va * vb; break;\n"
    "    case 3: r = va / vb; break;\n"
    "    case 4: r = vb * va * (1.0 - va); break;\n"
    "    case 5: r = vb * (1.0 - va * va); break;\n"
    "    case 6: r = va > 0.0 ? vb : 0.0; break;\n"
    "    default: r = va; break;\n"
    "    }\n"
    "    out[tid] = r;\n"
    "}\n"
    "\n"
    "/* ---- accumulate gather ---- */\n"
    "kernel void pg_k_accum_gather(\n"
    "    device float *dst [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant float &scale [[buffer(2)]],\n"
    "    constant Strides &ar [[buffer(3)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (tid >= ar.numel) return;\n"
    "    u32 idx[8];\n"
    "    u32 rem = tid;\n"
    "    for (u32 d = ar.ndim; d-- > 0;) {\n"
    "        idx[d] = rem % ar.shape[d];\n"
    "        rem /= ar.shape[d];\n"
    "    }\n"
    "    u32 os = 0;\n"
    "    for (u32 d = 0; d < ar.ndim; d++)\n"
    "        os += idx[d] * ar.s[d];\n"
    "    dst[tid] += scale * src[os];\n"
    "}\n"
    "\n"
    "/* ---- accumulate scatter ---- */\n"
    "kernel void pg_k_accum_scatter(\n"
    "    device float *dst [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant float &scale [[buffer(2)]],\n"
    "    constant Strides &ar [[buffer(3)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (tid >= ar.numel) return;\n"
    "    u32 idx[8];\n"
    "    u32 rem = tid;\n"
    "    for (u32 d = ar.ndim; d-- > 0;) {\n"
    "        idx[d] = rem % ar.shape[d];\n"
    "        rem /= ar.shape[d];\n"
    "    }\n"
    "    u32 od = 0;\n"
    "    for (u32 d = 0; d < ar.ndim; d++)\n"
    "        od += idx[d] * ar.s[d];\n"
    "    dst[od] += scale * src[tid];\n"
    "}\n"
    "\n"
    "/* ---- sum over axis ---- */\n"
    "kernel void pg_k_sum_axis(\n"
    "    device float *out [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant float &scale [[buffer(2)]],\n"
    "    constant u32 &outer [[buffer(3)]],\n"
    "    constant u32 &len [[buffer(4)]],\n"
    "    constant u32 &inner [[buffer(5)]],\n"
    "    constant u32 &ostride [[buffer(6)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    u32 total = outer * inner;\n"
    "    if (tid >= total) return;\n"
    "    u32 o = tid / inner;\n"
    "    u32 ii = tid % inner;\n"
    "    float acc = 0.0;\n"
    "    device const float *p = src + o * len * inner + ii;\n"
    "    for (u32 j = 0; j < len; j++, p += inner)\n"
    "        acc += *p;\n"
    "    out[o * ostride + ii] = acc * scale;\n"
    "}\n"
    "\n"
    "/* ---- softmax ---- */\n"
    "kernel void pg_k_softmax(\n"
    "    device float *out [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant u32 &outer [[buffer(2)]],\n"
    "    constant u32 &len [[buffer(3)]],\n"
    "    constant u32 &inner [[buffer(4)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    u32 total = outer * inner;\n"
    "    if (tid >= total) return;\n"
    "    u32 o = tid / inner;\n"
    "    u32 ii = tid % inner;\n"
    "    device const float *col = src + o * len * inner + ii;\n"
    "    device float *dcol = out + o * len * inner + ii;\n"
    "    float mx = col[0];\n"
    "    for (u32 j = 1; j < len; j++) {\n"
    "        float v = col[j * inner];\n"
    "        if (v > mx) mx = v;\n"
    "    }\n"
    "    float sum = 0.0;\n"
    "    for (u32 j = 0; j < len; j++) {\n"
    "        float e = exp(col[j * inner] - mx);\n"
    "        dcol[j * inner] = e;\n"
    "        sum += e;\n"
    "    }\n"
    "    float inv = 1.0 / sum;\n"
    "    for (u32 j = 0; j < len; j++)\n"
    "        dcol[j * inner] *= inv;\n"
    "}\n"
    "\n"
    "/* ---- strided copy ---- */\n"
    "kernel void pg_k_copy_strided(\n"
    "    device float *dst [[buffer(0)]],\n"
    "    device const float *src [[buffer(1)]],\n"
    "    constant Strides &ar [[buffer(2)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (tid >= ar.numel) return;\n"
    "    u32 idx[8];\n"
    "    u32 rem = tid;\n"
    "    for (u32 d = ar.ndim; d-- > 0;) {\n"
    "        idx[d] = rem % ar.shape[d];\n"
    "        rem /= ar.shape[d];\n"
    "    }\n"
    "    u32 os = 0;\n"
    "    for (u32 d = 0; d < ar.ndim; d++)\n"
    "        os += idx[d] * ar.s[d];\n"
    "    dst[tid] = src[os];\n"
    "}\n"
    "\n"
    "/* ---- tiled SGEMM (64x64, BK=16, 4x4 micro-tile) ---- */\n"
    "#define BM 64\n"
    "#define BN 64\n"
    "#define BK 16\n"
    "#define TP 16\n"
    "\n"
    "kernel void pg_k_sgemm(\n"
    "    device const float *a [[buffer(0)]],\n"
    "    device const float *b [[buffer(1)]],\n"
    "    device float *c [[buffer(2)]],\n"
    "    constant u32 &M [[buffer(3)]],\n"
    "    constant u32 &N [[buffer(4)]],\n"
    "    constant u32 &K [[buffer(5)]],\n"
    "    uint2 tid [[thread_position_in_threadgroup]],\n"
    "    uint2 bid [[threadgroup_position_in_grid]],\n"
    "    uint2 tpg [[threads_per_threadgroup]])\n"
    "{\n"
    "    const u32 tx = tid.x;\n"
    "    const u32 ty = tid.y;\n"
    "    const u32 tid_lin = ty * TP + tx;\n"
    "\n"
    "    threadgroup float s_a[BM * BK];\n"
    "    threadgroup float s_b[BK * BN];\n"
    "\n"
    "    const u32 row0 = bid.y * BM + ty * 4;\n"
    "    const u32 col0 = bid.x * BN + tx * 4;\n"
    "\n"
    "    float acc[4][4];\n"
    "    for (u32 i = 0; i < 4; i++)\n"
    "        for (u32 j = 0; j < 4; j++)\n"
    "            acc[i][j] = 0.0;\n"
    "\n"
    "    const u32 ntiles = (K + BK - 1) / BK;\n"
    "\n"
    "    for (u32 t = 0; t < ntiles; t++) {\n"
    "        const u32 gk = t * BK;\n"
    "\n"
    "        for (u32 idx = tid_lin; idx < BM * BK; idx += TP * TP) {\n"
    "            u32 r = idx / BK;\n"
    "            u32 cc = idx % BK;\n"
    "            u32 gr = bid.y * BM + r;\n"
    "            u32 gc = gk + cc;\n"
    "            s_a[idx] = (gr < M && gc < K) ? a[gr * K + gc] : 0.0;\n"
    "        }\n"
    "        for (u32 idx = tid_lin; idx < BK * BN; idx += TP * TP) {\n"
    "            u32 rr = idx / BN;\n"
    "            u32 cc = idx % BN;\n"
    "            u32 gr = gk + rr;\n"
    "            u32 gc = bid.x * BN + cc;\n"
    "            s_b[idx] = (gr < K && gc < N) ? b[gr * N + gc] : 0.0;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        for (u32 kk = 0; kk < BK; kk++) {\n"
    "            float ra[4], rb[4];\n"
    "            for (u32 i = 0; i < 4; i++)\n"
    "                ra[i] = s_a[(ty * 4 + i) * BK + kk];\n"
    "            for (u32 j = 0; j < 4; j++)\n"
    "                rb[j] = s_b[kk * BN + tx * 4 + j];\n"
    "            for (u32 i = 0; i < 4; i++)\n"
    "                for (u32 j = 0; j < 4; j++)\n"
    "                    acc[i][j] += ra[i] * rb[j];\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    for (u32 i = 0; i < 4; i++) {\n"
    "        u32 gr = row0 + i;\n"
    "        if (gr >= M) continue;\n"
    "        for (u32 j = 0; j < 4; j++) {\n"
    "            u32 gc = col0 + j;\n"
    "            if (gc < N)\n"
    "                c[gr * N + gc] = acc[i][j];\n"
    "        }\n"
    "    }\n"
    "}\n"
    ;
}

/* ---------- compile MSL library, look up kernels ---------- */

static pg_status metal_compile_library(void)
{
    @autoreleasepool {
        NSString *src = [NSString stringWithUTF8String:msl_source()];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:src
                                                   options:nil
                                                     error:&err];
        if (!lib) {
            fprintf(stderr, "[metal] MSL compile error: %s\n",
                    [[err localizedDescription] UTF8String]);
            return PG_ERR_GEMM;
        }

        #define LOAD(name, fn) do { \
            id<MTLFunction> f = [lib newFunctionWithName:@#name]; \
            if (!f) { fprintf(stderr, "[metal] kernel not found: %s\n", #name); return PG_ERR_GEMM; } \
            fn = [g_device newComputePipelineStateWithFunction:f error:&err]; \
            if (!fn) { fprintf(stderr, "[metal] pipeline error: %s\n", #name); return PG_ERR_GEMM; } \
        } while(0)

        LOAD(pg_k_sgemm,        g_p_sgemm);
        LOAD(pg_k_map,           g_p_map);
        LOAD(pg_k_bin,           g_p_bin);
        LOAD(pg_k_accum_gather,  g_p_accum_gather);
        LOAD(pg_k_accum_scatter, g_p_accum_scatter);
        LOAD(pg_k_sum_axis,      g_p_sum_axis);
        LOAD(pg_k_softmax,       g_p_softmax);
        LOAD(pg_k_copy_strided,  g_p_copy_strided);

        #undef LOAD
    }
    return PG_OK;
}

/* ---------- backend ops ---------- */

static pg_status metal_init(void)
{
    static pg_status cached = PG_ERR_UNSUPPORTED;
    static int done = 0;
    if (done) return cached;
    done = 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[metal] no Metal device\n");
            return cached;
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "[metal] failed to create command queue\n");
            return cached;
        }
    }

    pg_status st = metal_compile_library();
    if (st != PG_OK) return cached = st;

    cached = PG_OK;
    return cached;
}

static void *metal_malloc(size_t nbytes)
{
    if (metal_init() != PG_OK) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_device newBufferWithLength:nbytes ? nbytes : 1
                                                 options:MTLResourceStorageModeShared];
        return buf ? [buf contents] : NULL;
    }
}

static void metal_free(void *p)
{
    (void)p;
}

static pg_status metal_h2d(void *dst, const void *src, size_t nbytes)
{
    if (metal_init() != PG_OK) return PG_ERR_COPY;
    memcpy(dst, src, nbytes);
    return PG_OK;
}

static pg_status metal_d2h(void *dst, const void *src, size_t nbytes)
{
    if (metal_init() != PG_OK) return PG_ERR_COPY;
    memcpy(dst, src, nbytes);
    return PG_OK;
}

static pg_status metal_sync(void)
{
    if (metal_init() != PG_OK) return PG_ERR_SYNC;
    @autoreleasepool {
        id<MTLCommandBuffer> buf = [g_queue commandBuffer];
        [buf commit];
        [buf waitUntilCompleted];
    }
    return PG_OK;
}

/* ---------- GEMM ---------- */

#define BTILE 64
#define TPT   16

static void metal_gemm(size_t m, size_t n, size_t k,
                       const float *a, size_t lda,
                       const float *b, size_t ldb,
                       float *c, size_t ldc)
{
    assert(lda == k && ldb == n && ldc == n);
    if (metal_init() != PG_OK) { assert(!"metal backend not initialized"); return; }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:g_p_sgemm];
        [enc setBuffer:(void *)a  offset:0 atIndex:0];
        [enc setBuffer:(void *)b  offset:0 atIndex:1];
        [enc setBuffer:c           offset:0 atIndex:2];

        uint32_t m32 = (uint32_t)m, n32 = (uint32_t)n, k32 = (uint32_t)k;
        [enc setBytes:&m32 length:sizeof(m32) atIndex:3];
        [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
        [enc setBytes:&k32 length:sizeof(k32) atIndex:5];

        MTLSize grid  = MTLSizeMake((n + BTILE - 1) / BTILE * TPT,
                                    (m + BTILE - 1) / BTILE * TPT, 1);
        MTLSize group = MTLSizeMake(TPT, TPT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

/* ---------- GPU kernel dispatch helpers ---------- */

static pg_status metal_dispatch_1d(id<MTLComputePipelineState> pso,
                                   size_t n, size_t nargs,
                                   void (^set_args)(id<MTLComputeCommandEncoder>))
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        set_args(enc);

        NSUInteger tg = [pso threadExecutionWidth];
        MTLSize grid  = MTLSizeMake((n + tg - 1) / tg, 1, 1);
        MTLSize group = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return PG_OK;
}

static pg_status metal_gpu_map(float *out, const float *src, size_t n, int op)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    uint32_t n32 = (uint32_t)n, op32 = (uint32_t)op;
    return metal_dispatch_1d(g_p_map, n, 4, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)out  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
        [enc setBytes:&op32 length:sizeof(op32) atIndex:3];
    });
}

static pg_status metal_gpu_bin(float *out, const float *a, const float *b,
                                size_t n, int op, const pg_k_bin_args *args)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    uint32_t op32 = (uint32_t)op;
    return metal_dispatch_1d(g_p_bin, n, 5, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)out  offset:0 atIndex:0];
        [enc setBuffer:(void *)a    offset:0 atIndex:1];
        [enc setBuffer:(void *)b    offset:0 atIndex:2];
        [enc setBytes:&op32 length:sizeof(op32) atIndex:3];
        [enc setBytes:(void *)args length:sizeof(*args) atIndex:4];
    });
}

static pg_status metal_gpu_accum_gather(float *dst, const float *src,
                                         float scale, const pg_k_strides *args)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    return metal_dispatch_1d(g_p_accum_gather, args->numel, 4, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)dst  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc setBytes:(void *)args length:sizeof(*args) atIndex:3];
    });
}

static pg_status metal_gpu_accum_scatter(float *dst, const float *src,
                                          float scale, const pg_k_strides *args)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    return metal_dispatch_1d(g_p_accum_scatter, args->numel, 4, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)dst  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc setBytes:(void *)args length:sizeof(*args) atIndex:3];
    });
}

static pg_status metal_gpu_sum_axis(float *out, const float *src, float scale,
                                     size_t outer, size_t len, size_t inner,
                                     size_t keepdim_stride)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    uint32_t o32 = (uint32_t)outer, l32 = (uint32_t)len;
    uint32_t i32 = (uint32_t)inner, ks32 = (uint32_t)keepdim_stride;
    return metal_dispatch_1d(g_p_sum_axis, outer * inner, 7, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)out  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc setBytes:&o32  length:sizeof(o32)  atIndex:3];
        [enc setBytes:&l32  length:sizeof(l32)  atIndex:4];
        [enc setBytes:&i32  length:sizeof(i32)  atIndex:5];
        [enc setBytes:&ks32 length:sizeof(ks32) atIndex:6];
    });
}

static pg_status metal_gpu_softmax(float *out, const float *src,
                                    size_t outer, size_t len, size_t inner)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    uint32_t o32 = (uint32_t)outer, l32 = (uint32_t)len, i32 = (uint32_t)inner;
    return metal_dispatch_1d(g_p_softmax, outer * inner, 5, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)out  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:&o32 length:sizeof(o32) atIndex:2];
        [enc setBytes:&l32 length:sizeof(l32) atIndex:3];
        [enc setBytes:&i32 length:sizeof(i32) atIndex:4];
    });
}

static pg_status metal_gpu_copy_strided(float *dst, const float *src,
                                         const pg_k_strides *args)
{
    if (metal_init() != PG_OK) return PG_ERR_GEMM;
    return metal_dispatch_1d(g_p_copy_strided, args->numel, 3, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:(void *)dst  offset:0 atIndex:0];
        [enc setBuffer:(void *)src  offset:0 atIndex:1];
        [enc setBytes:(void *)args length:sizeof(*args) atIndex:2];
    });
}

static pg_status metal_gpu_fill(void *p, size_t nbytes, float v)
{
    if (metal_init() != PG_OK) return PG_ERR_COPY;
    float *fp = (float *)p;
    size_t nf = nbytes / sizeof(float);
    for (size_t i = 0; i < nf; i++)
        fp[i] = v;
    return PG_OK;
}

/* ---------- register ---------- */

static void metal_register_gpu(void)
{
    pg_gpu.map          = metal_gpu_map;
    pg_gpu.bin          = metal_gpu_bin;
    pg_gpu.accum_gather  = metal_gpu_accum_gather;
    pg_gpu.accum_scatter = metal_gpu_accum_scatter;
    pg_gpu.sum_axis     = metal_gpu_sum_axis;
    pg_gpu.softmax      = metal_gpu_softmax;
    pg_gpu.copy_strided = metal_gpu_copy_strided;
    pg_gpu.fill         = metal_gpu_fill;
    pg_gpu.copy_d2d     = NULL;
}

static pg_status metal_init_and_register(void)
{
    pg_status st = metal_init();
    if (st == PG_OK)
        metal_register_gpu();
    return st;
}

const pg_backend_ops pg_backend_metal = {
    .name = "metal",
    .init = metal_init_and_register,
    .malloc = metal_malloc,
    .free = metal_free,
    .copy_h2d = metal_h2d,
    .copy_d2h = metal_d2h,
    .sync = metal_sync,
    .gemm = metal_gemm,
};

#else /* !__APPLE__ */

static pg_status metal_init(void)
{
    return PG_ERR_UNSUPPORTED;
}

static void *metal_malloc(size_t nbytes)
{
    (void)nbytes;
    return NULL;
}

static void metal_free(void *p)
{
    (void)p;
}

static pg_status metal_h2d(void *dst, const void *src, size_t nbytes)
{
    (void)dst; (void)src; (void)nbytes;
    return PG_ERR_UNSUPPORTED;
}

static pg_status metal_d2h(void *dst, const void *src, size_t nbytes)
{
    (void)dst; (void)src; (void)nbytes;
    return PG_ERR_UNSUPPORTED;
}

static pg_status metal_sync(void)
{
    return PG_ERR_UNSUPPORTED;
}

static void metal_gemm(size_t m, size_t n, size_t k,
                       const float *a, size_t lda,
                       const float *b, size_t ldb,
                       float *c, size_t ldc)
{
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)b; (void)ldb; (void)c; (void)ldc;
}

const pg_backend_ops pg_backend_metal = {
    .name = "metal",
    .init = metal_init,
    .malloc = metal_malloc,
    .free = metal_free,
    .copy_h2d = metal_h2d,
    .copy_d2h = metal_d2h,
    .sync = metal_sync,
    .gemm = metal_gemm,
};

#endif /* __APPLE__ */

#endif /* PICOGRAD_BACKEND_METAL */
