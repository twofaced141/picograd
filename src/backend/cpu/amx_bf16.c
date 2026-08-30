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
    // TODO: implement with _tile_loadd, _tile_dpbf16ps
    // For now, scalar fallback with bf16 conversion
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
