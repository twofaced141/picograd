#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../core/convert.h"
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

// AMX-BF16: tileloadd/tdpbf16ps 16x32 tile - 16x vs AVX512
// Requires AMX_TILE + __builtin_cpu_supports("amx-bf16")
// Placeholder: uses generic fallback until AMX hardware available.
// Weights prepack in AMX format offline (not implemented here).

bool pg_cpu_supports_amx_bf16(void){
#if defined(__x86_64__) && !defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("amx-bf16") && __builtin_cpu_supports("amx-tile");
#else
    return false;
#endif
}

void amx_bgemm_micro(size_t k, const uint16_t *a, size_t lda,
                     const uint16_t *b, size_t ldb,
                     float *c, size_t ldc, size_t m, size_t n){
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    if(m==16 && n==16 && k==32 && pg_cpu_supports_amx_bf16()){
        // AMX tile config: 16x32 BF16 A, 32x16 BF16 B, 16x16 FP32 C
        // Palette 1, 1KB tiles (16 rows *64 bytes)
        struct __tile_config { uint8_t palette_id; uint8_t start_row; uint8_t reserved[14]; uint16_t colsb[8]; uint8_t rows[8]; } cfg={0};
        cfg.palette_id=1;
        cfg.colsb[0]=64; cfg.rows[0]=16; // C f32 16x16
        cfg.colsb[1]=64; cfg.rows[1]=16; // A bf16 16x32
        cfg.colsb[2]=64; cfg.rows[2]=32; // B bf16 32x16 (transposed layout)
        _tile_loadconfig(&cfg);
        _tile_loadd(0, c, ldc*sizeof(float));
        _tile_loadd(1, a, lda*2);
        _tile_loadd(2, b, ldb*2);
        _tile_dpbf16ps(0, 1, 2);
        _tile_stored(0, c, ldc*sizeof(float));
        _tile_release();
        return;
    }
#endif
    for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++){ float acc=c[i*ldc+j]; for(size_t p=0;p<k;p++) acc+=pg_bf16_to_f32_scalar(a[i*lda+p])*pg_bf16_to_f32_scalar(b[p*ldb+j]); c[i*ldc+j]=acc; }
}

void bgemm_amx_micro(size_t k, const uint16_t *a, size_t lda,
                     const uint16_t *b, size_t ldb,
                     float *c, size_t ldc, size_t m, size_t n){
    if (pg_cpu_supports_amx_bf16()){
        amx_bgemm_micro(k,a,lda,b,ldb,c,ldc,m,n);
    } else {
        for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++){ float acc=c[i*ldc+j]; for(size_t p=0;p<k;p++) acc+=pg_bf16_to_f32_scalar(a[i*lda+p])*pg_bf16_to_f32_scalar(b[p*ldb+j]); c[i*ldc+j]=acc; }
    }
}
