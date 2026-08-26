.intel_syntax noprefix

.text
.p2align 5
.globl sgemm_avx512_micro
.type sgemm_avx512_micro, @function

# void sgemm_avx512_micro(size_t k,
#                         const float *a, size_t lda,
#                         const float *b, size_t ldb,
#                         float *c, size_t ldc,
#                         size_t m, size_t n)
#
# 8x16 microkernel: C[0..8)x[0..16) += A[0..8)xk * B[k x[0..16)
# rdi=k rsi=a rdx=lda rcx=b r8=ldb r9=c [rsp+56]=ldc [rsp+64]=m [rsp+72]=n
sgemm_avx512_micro:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r15, rsi
    mov     r13, r8
    shl     r13, 2
    shl     rdx, 2

    mov     rbp, [rsp + 72]
    shl     rbp, 3
    lea     rax, [rip + .Lcol_masks]
    kmovq   k1, [rax + rbp]

    mov     rsi, r15
    lea     r8,  [r15 + rdx]
    lea     r10, [r15 + rdx*2]
    lea     r11, [r15 + rdx*2]
    add     r11, rdx
    lea     rax, [r15 + rdx*4]
    lea     rbx, [rax + rdx]
    lea     r12, [rax + rdx*2]
    lea     r14, [rax + rdx*2]
    add     r14, rdx

    cmp     qword ptr [rsp + 64], 1
    cmovbe  r8,  r15
    cmp     qword ptr [rsp + 64], 2
    cmovbe  r10, r15
    cmp     qword ptr [rsp + 64], 3
    cmovbe  r11, r15
    cmp     qword ptr [rsp + 64], 4
    cmovbe  rax, r15
    cmp     qword ptr [rsp + 64], 5
    cmovbe  rbx, r15
    cmp     qword ptr [rsp + 64], 6
    cmovbe  r12, r15
    cmp     qword ptr [rsp + 64], 7
    cmovbe  r14, r15

    mov     rdx, [rsp + 56]
    shl     rdx, 2

    vmovups zmm0 {k1}{z}, [r9]

    lea     rbp, [r9 + rdx]
    cmp     qword ptr [rsp + 64], 1
    jbe     1f
    vmovups zmm1 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm1, zmm1, zmm1
2:
    lea     rbp, [r9 + rdx*2]
    cmp     qword ptr [rsp + 64], 2
    jbe     1f
    vmovups zmm2 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm2, zmm2, zmm2
2:
    lea     rbp, [r9 + rdx*2]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 3
    jbe     1f
    vmovups zmm3 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm3, zmm3, zmm3
2:
    lea     rbp, [r9 + rdx*4]
    cmp     qword ptr [rsp + 64], 4
    jbe     1f
    vmovups zmm4 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm4, zmm4, zmm4
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 5
    jbe     1f
    vmovups zmm5 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm5, zmm5, zmm5
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 6
    jbe     1f
    vmovups zmm6 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm6, zmm6, zmm6
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 7
    jbe     1f
    vmovups zmm7 {k1}{z}, [rbp]
    jmp     2f
1:  vpxord  zmm7, zmm7, zmm7
2:

    test    rdi, rdi
    jz      .Lepilogue

.p2align 5
.Lkloop:
    vmovups     zmm8 {k1}{z}, [rcx]
    add         rcx, r13
    vbroadcastss zmm9, [rsi]
    vfmadd231ps zmm0, zmm9, zmm8
    vbroadcastss zmm9, [r8]
    vfmadd231ps zmm1, zmm9, zmm8
    vbroadcastss zmm9, [r10]
    vfmadd231ps zmm2, zmm9, zmm8
    vbroadcastss zmm9, [r11]
    vfmadd231ps zmm3, zmm9, zmm8
    vbroadcastss zmm9, [rax]
    vfmadd231ps zmm4, zmm9, zmm8
    vbroadcastss zmm9, [rbx]
    vfmadd231ps zmm5, zmm9, zmm8
    vbroadcastss zmm9, [r12]
    vfmadd231ps zmm6, zmm9, zmm8
    vbroadcastss zmm9, [r14]
    vfmadd231ps zmm7, zmm9, zmm8
    add     rsi, 4
    add     r8,  4
    add     r10, 4
    add     r11, 4
    add     rax, 4
    add     rbx, 4
    add     r12, 4
    add     r14, 4
    dec     rdi
    jnz     .Lkloop

.Lepilogue:
    vmovups [r9] {k1}, zmm0

    lea     rbp, [r9 + rdx]
    cmp     qword ptr [rsp + 64], 1
    jbe     1f
    vmovups [rbp] {k1}, zmm1
1:
    lea     rbp, [r9 + rdx*2]
    cmp     qword ptr [rsp + 64], 2
    jbe     1f
    vmovups [rbp] {k1}, zmm2
1:
    lea     rbp, [r9 + rdx*2]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 3
    jbe     1f
    vmovups [rbp] {k1}, zmm3
1:
    lea     rbp, [r9 + rdx*4]
    cmp     qword ptr [rsp + 64], 4
    jbe     1f
    vmovups [rbp] {k1}, zmm4
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 5
    jbe     1f
    vmovups [rbp] {k1}, zmm5
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 6
    jbe     1f
    vmovups [rbp] {k1}, zmm6
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 7
    jbe     1f
    vmovups [rbp] {k1}, zmm7
1:

    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

.size sgemm_avx512_micro, .-sgemm_avx512_micro

.section .rodata
.p2align 3
.Lcol_masks:
    .quad 0x0000000000000000
    .quad 0x0000000000000001
    .quad 0x0000000000000003
    .quad 0x0000000000000007
    .quad 0x000000000000000f
    .quad 0x000000000000001f
    .quad 0x000000000000003f
    .quad 0x000000000000007f
    .quad 0x00000000000000ff
    .quad 0x00000000000001ff
    .quad 0x00000000000003ff
    .quad 0x00000000000007ff
    .quad 0x0000000000000fff
    .quad 0x0000000000001fff
    .quad 0x0000000000003fff
    .quad 0x0000000000007fff
    .quad 0x000000000000ffff

.section .note.GNU-stack, "", @progbits
