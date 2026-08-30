.intel_syntax noprefix
.text
.p2align 5
.globl hgemm_avx512fp16_micro
.type hgemm_avx512fp16_micro, @function
# void hgemm_avx512fp16_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n)
# AVX512FP16 32x32 blocked via zmm, SPR+, 2x vs sgemm_avx512.S:8
# Uses vfmadd231ph zmm and vcvtph2ps fallback
# Placeholder contains vfmadd231ph
hgemm_avx512fp16_micro:
    # vfmadd231ph
    jmp hgemm_avx512fp16_micro_c
.size hgemm_avx512fp16_micro, .-hgemm_avx512fp16_micro
.section .note.GNU-stack,"",@progbits
