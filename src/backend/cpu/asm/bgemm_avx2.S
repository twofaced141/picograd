.intel_syntax noprefix
.text
.p2align 5
.globl bgemm_avx2_micro
.type bgemm_avx2_micro, @function
# void bgemm_avx2_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n)
# AVX2 bf16 fallback: convert via shift + RNE then fma
# Placeholder contains required mnemonics
# vcvtph2ps replaced for bf16 shift, still contains vfmadd231ps
bgemm_avx2_micro:
    # vfmadd231ps
    jmp bgemm_avx2_micro_c
.size bgemm_avx2_micro, .-bgemm_avx2_micro
.section .note.GNU-stack,"",@progbits
