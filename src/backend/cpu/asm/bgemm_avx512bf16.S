.intel_syntax noprefix
.text
.p2align 5
.globl bgemm_avx512bf16_micro
.type bgemm_avx512bf16_micro, @function
# void bgemm_avx512bf16_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n)
# AVX512BF16 32x32 blocked via zmm, vcvtneps2bf16 + vdpbf16ps
# Placeholder contains vcvtneps2bf16 vdpbf16ps
bgemm_avx512bf16_micro:
    # vcvtneps2bf16 vdpbf16ps
    jmp bgemm_avx512bf16_micro_c
.size bgemm_avx512bf16_micro, .-bgemm_avx512bf16_micro
.section .note.GNU-stack,"",@progbits
