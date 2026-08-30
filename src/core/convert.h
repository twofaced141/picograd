#ifndef PICOGRAD_CORE_CONVERT_H
#define PICOGRAD_CORE_CONVERT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// scalar bit-hacks ---------------------------------------------------------

// f32 -> f16  (IEEE 754, RNE, handles Inf/NaN)
static inline uint16_t pg_f32_to_f16_scalar(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x007FFFFFu;

    if (exp == 255) { // Inf / NaN
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        // preserve some payload, quiet NaN
        uint16_t nan = (uint16_t)(sign | 0x7C00u | (mant >> 13));
        if ((nan & 0x03FFu) == 0) nan |= 1;
        return nan | 0x0200u;
    }
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u); // overflow -> Inf
    if (e <= 0) {
        // subnormal or underflow
        if (e < -10) return (uint16_t)sign; // flush to zero
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(1 - e);
        // RNE rounding: add 1<<(shift-1) with tie-to-even
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t extra = mant & ((1u << shift) - 1u);
        mant >>= shift;
        // rounding
        if (extra > round_bit || (extra == round_bit && (mant & 1u))) mant++;
        return (uint16_t)(sign | mant);
    }
    // normal
    // RNE: add 0xFFF + ((mant>>13)&1) before truncating 13 bits
    uint32_t rounding = 0x0FFFu + ((mant >> 13) & 1u);
    mant += rounding;
    // mant overflow may increment exp
    if (mant & 0x00800000u) { mant = 0; e++; }
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)e << 10) | (mant >> 13));
}

static inline float pg_f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = ((uint32_t)h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal -> normalized
            // normalize mantissa
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            uint32_t e = (uint32_t)((int32_t)exp + (127 - 15));
            f = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf / NaN
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t e = exp + (127 - 15);
        f = sign | (e << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// f32 -> bf16 : truncate with RNE (round to nearest even)
// ref: bf16 = (f32 bits + 0x7FFF + lsb) >>16
static inline uint16_t pg_f32_to_bf16_scalar(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    // NaN preserve
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu) != 0) {
        // NaN -> quiet NaN bf16
        return (uint16_t)(0x7FC0u | ((x >> 16) & 0x1u));
    }
    uint32_t rounding_bias = 0x00007FFFu + ((x >> 16) & 1u);
    // avoid overflow for Inf? Inf stays Inf
    uint32_t y = x + rounding_bias;
    return (uint16_t)(y >> 16);
}

static inline float pg_bf16_to_f32_scalar(uint16_t h) {
    uint32_t x = ((uint32_t)h) << 16;
    float out;
    memcpy(&out, &x, sizeof(out));
    return out;
}

// batch converters ---------------------------------------------------------

// scalar fallbacks used when AVX not available
static inline void pg_convert_f16_to_f32_scalar_batch(float *dst, const uint16_t *src, size_t n){
    for(size_t i=0;i<n;i++) dst[i]=pg_f16_to_f32_scalar(src[i]);
}
static inline void pg_convert_f32_to_f16_scalar_batch(uint16_t *dst, const float *src, size_t n){
    for(size_t i=0;i<n;i++) dst[i]=pg_f32_to_f16_scalar(src[i]);
}
static inline void pg_convert_bf16_to_f32_scalar_batch(float *dst, const uint16_t *src, size_t n){
    for(size_t i=0;i<n;i++) dst[i]=pg_bf16_to_f32_scalar(src[i]);
}
static inline void pg_convert_f32_to_bf16_scalar_batch(uint16_t *dst, const float *src, size_t n){
    for(size_t i=0;i<n;i++) dst[i]=pg_f32_to_bf16_scalar(src[i]);
}

// accelerated batch using intrinsics when available
#if defined(__x86_64__) || defined(__i386__) || defined(PG_ARCH_X86_64)
#if defined(__AVX512FP16__) || defined(__AVX512BF16__) || defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#endif

static inline void pg_f32_to_f16_batch(uint16_t *dst, const float *src, size_t n){
#if defined(__F16C__) && (defined(__AVX2__) || defined(__AVX__))
    // use _mm256_cvtps_ph + _mm_cvtps_ph for tails if available (runtime fallback is scalar)
    // We keep scalar for correctness; accelerated path only when caller knows F16C is available.
    // To avoid illegal instruction, we check at runtime when possible via builtin.
    // For simplicity, use scalar loop unless compiled with -mf16c and runtime supports it.
    // The library will call scalar; x86 microkernels will use inline asm vcvtph2ps directly.
    (void)dst; (void)src; (void)n;
#endif
    pg_convert_f32_to_f16_scalar_batch(dst, src, n);
}

static inline void pg_f16_to_f32_batch(float *dst, const uint16_t *src, size_t n){
    pg_convert_f16_to_f32_scalar_batch(dst, src, n);
}

static inline void pg_f32_to_bf16_batch(uint16_t *dst, const float *src, size_t n){
    pg_convert_f32_to_bf16_scalar_batch(dst, src, n);
}
static inline void pg_bf16_to_f32_batch(float *dst, const uint16_t *src, size_t n){
    pg_convert_bf16_to_f32_scalar_batch(dst, src, n);
}

#ifdef __cplusplus
}
#endif

#endif
