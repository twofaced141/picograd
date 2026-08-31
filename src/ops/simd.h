#ifndef PICOGRAD_OPS_SIMD_H
#define PICOGRAD_OPS_SIMD_H

#include <stddef.h>

// Determine arch via Makefile PG_ARCH_* or auto-detect
#if defined(PG_ARCH_X86_64)
  #if defined(__AVX2__) || defined(__AVX__)
    #define PG_SIMD_AVX 1
  #else
    // x86 arch but AVX not enabled (generic fallback)
    #define PG_SIMD_GENERIC 1
  #endif
#elif defined(PG_ARCH_AARCH64)
  #define PG_SIMD_NEON 1
#elif defined(PG_ARCH_GENERIC)
  #define PG_SIMD_GENERIC 1
#else
  // auto-detect when PG_ARCH not defined (direct cc invocation)
  #if defined(__AVX2__) || defined(__AVX__)
    #define PG_SIMD_AVX 1
  #elif defined(__aarch64__)
    #define PG_SIMD_NEON 1
  #else
    #define PG_SIMD_GENERIC 1
  #endif
#endif

#if defined(PG_SIMD_AVX)
#include <immintrin.h>

static inline void simd_bin_add(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] + b[i];
}
static inline void simd_bin_sub(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] - b[i];
}
static inline void simd_bin_mul(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] * b[i];
}
static inline void simd_bin_div(const float *a, const float *b, float *o, size_t n){
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] / b[i];
}
static inline void simd_scalar_add(const float *a, float bv, float *o, size_t n){
    __m256 vb = _mm256_set1_ps(bv);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(o + i, vc);
    }
    for (; i < n; i++) o[i] = a[i] + bv;
}
static inline void simd_relu(float *d, size_t n){
    size_t i = 0;
    __m256 z = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(d + i);
        v = _mm256_max_ps(v, z);
        _mm256_storeu_ps(d + i, v);
    }
    for (; i < n; i++) d[i] = d[i] > 0 ? d[i] : 0;
}
static inline void simd_leaky_relu(float *d, size_t n, float alpha){
    size_t i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    __m256 z = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(d + i);
        __m256 mask = _mm256_cmp_ps(v, z, _CMP_GT_OQ);
        __m256 av = _mm256_mul_ps(v, a);
        __m256 res = _mm256_blendv_ps(av, v, mask);
        _mm256_storeu_ps(d + i, res);
    }
    for (; i < n; i++) {
        float v = d[i];
        d[i] = v > 0 ? v : alpha * v;
    }
}

#elif defined(PG_SIMD_NEON)
#include "simd_neon.h"

static inline void simd_bin_add(const float *a, const float *b, float *o, size_t n) {
    simd_neon_bin_add(a, b, o, n);
}
static inline void simd_bin_sub(const float *a, const float *b, float *o, size_t n) {
    simd_neon_bin_sub(a, b, o, n);
}
static inline void simd_bin_mul(const float *a, const float *b, float *o, size_t n) {
    simd_neon_bin_mul(a, b, o, n);
}
static inline void simd_bin_div(const float *a, const float *b, float *o, size_t n) {
    simd_neon_bin_div(a, b, o, n);
}
static inline void simd_scalar_add(const float *a, float bv, float *o, size_t n) {
    simd_neon_scalar_add(a, bv, o, n);
}
static inline void simd_relu(float *d, size_t n) {
    simd_neon_relu(d, n);
}
static inline void simd_leaky_relu(float *d, size_t n, float alpha) {
    simd_neon_leaky_relu(d, n, alpha);
}

#else // generic or fallback
// generic scalar with autovectorization hints

static inline void simd_bin_add(const float *a, const float *b, float *o, size_t n){
#pragma GCC ivdep
    for(size_t i=0;i<n;i++) o[i]=a[i]+b[i];
}
static inline void simd_bin_sub(const float *a, const float *b, float *o, size_t n){
#pragma GCC ivdep
    for(size_t i=0;i<n;i++) o[i]=a[i]-b[i];
}
static inline void simd_bin_mul(const float *a, const float *b, float *o, size_t n){
#pragma GCC ivdep
    for(size_t i = 0; i < n; i++) o[i] = a[i] * b[i];
}
static inline void simd_bin_div(const float *a, const float *b, float *o, size_t n){
#pragma GCC ivdep
    for(size_t i=0;i<n;i++) o[i]=a[i]/b[i];
}
static inline void simd_scalar_add(const float *a, float bv, float *o, size_t n){
#pragma GCC ivdep
    for(size_t i=0;i<n;i++) o[i]=a[i]+bv;
}
static inline void simd_relu(float *d, size_t n){
#pragma GCC ivdep
    for(size_t i = 0; i < n; i++) d[i] = d[i] > 0 ? d[i] : 0;
}
static inline void simd_leaky_relu(float *d, size_t n, float alpha){
#pragma GCC ivdep
    for(size_t i = 0; i < n; i++) {
        float v = d[i];
        d[i] = v > 0 ? v : alpha * v;
    }
}

#endif

#endif // PICOGRAD_OPS_SIMD_H
