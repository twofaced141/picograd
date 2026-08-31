#ifndef PICOGRAD_OPS_SIMD_NEON_H
#define PICOGRAD_OPS_SIMD_NEON_H

#if defined(__aarch64__)
#include <stddef.h>
#include <arm_neon.h>

static inline void simd_neon_bin_add(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] + b[i];
}
static inline void simd_neon_bin_sub(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vsubq_f32(va, vb);
        vst1q_f32(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] - b[i];
}
static inline void simd_neon_bin_mul(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] * b[i];
}
static inline void simd_neon_bin_div(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vdivq_f32(va, vb);
        vst1q_f32(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] / b[i];
}
static inline void simd_neon_scalar_add(const float *a, float bv, float *o, size_t n){
    float32x4_t vb = vdupq_n_f32(bv);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] + bv;
}

static inline void simd_neon_relu(float *d, size_t n){
    float32x4_t z = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(d + i);
        v = vmaxq_f32(v, z);
        vst1q_f32(d + i, v);
    }
    for (; i < n; i++) d[i] = d[i] > 0 ? d[i] : 0;
}
static inline void simd_neon_leaky_relu(float *d, size_t n, float alpha){
    float32x4_t a = vdupq_n_f32(alpha);
    float32x4_t z = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(d + i);
        uint32x4_t mask = vcgtq_f32(v, z);
        float32x4_t av = vmulq_f32(v, a);
        float32x4_t res = vbslq_f32(mask, v, av);
        vst1q_f32(d + i, res);
    }
    for (; i < n; i++) {
        float v = d[i];
        d[i] = v > 0 ? v : alpha * v;
    }
}

#endif // __aarch64__
#endif // PICOGRAD_OPS_SIMD_NEON_H
