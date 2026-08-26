.intel_syntax noprefix

.text
.p2align 5
.globl sgemm_avx2_micro
.type sgemm_avx2_micro, @function

sgemm_avx2_micro:
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
    shl     rbp, 5
    lea     rax, [rip + .Lcol_masks]
    vmovdqu ymm15, [rax + rbp]

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

    vmaskmovps ymm0, ymm15, [r9]

    lea     rbp, [r9 + rdx]
    cmp     qword ptr [rsp + 64], 1
    jbe     1f
    vmaskmovps ymm1, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm1, ymm1, ymm1
2:
    lea     rbp, [r9 + rdx*2]
    cmp     qword ptr [rsp + 64], 2
    jbe     1f
    vmaskmovps ymm2, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm2, ymm2, ymm2
2:
    lea     rbp, [r9 + rdx*2]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 3
    jbe     1f
    vmaskmovps ymm3, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm3, ymm3, ymm3
2:
    lea     rbp, [r9 + rdx*4]
    cmp     qword ptr [rsp + 64], 4
    jbe     1f
    vmaskmovps ymm4, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm4, ymm4, ymm4
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 5
    jbe     1f
    vmaskmovps ymm5, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm5, ymm5, ymm5
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 6
    jbe     1f
    vmaskmovps ymm6, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm6, ymm6, ymm6
2:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 7
    jbe     1f
    vmaskmovps ymm7, ymm15, [rbp]
    jmp     2f
1:  vxorps  ymm7, ymm7, ymm7
2:

    test    rdi, rdi
    jz      .Lepilogue

    cmp     qword ptr [rsp + 72], 8
    je      .Lkloop_full

.p2align 5
.Lkloop_tail:
    vmaskmovps  ymm8, ymm15, [rcx]
    add         rcx, r13
    vbroadcastss ymm9, [rsi]
    vfmadd231ps ymm0, ymm9, ymm8
    vbroadcastss ymm9, [r8]
    vfmadd231ps ymm1, ymm9, ymm8
    vbroadcastss ymm9, [r10]
    vfmadd231ps ymm2, ymm9, ymm8
    vbroadcastss ymm9, [r11]
    vfmadd231ps ymm3, ymm9, ymm8
    vbroadcastss ymm9, [rax]
    vfmadd231ps ymm4, ymm9, ymm8
    vbroadcastss ymm9, [rbx]
    vfmadd231ps ymm5, ymm9, ymm8
    vbroadcastss ymm9, [r12]
    vfmadd231ps ymm6, ymm9, ymm8
    vbroadcastss ymm9, [r14]
    vfmadd231ps ymm7, ymm9, ymm8
    add     rsi, 4
    add     r8,  4
    add     r10, 4
    add     r11, 4
    add     rax, 4
    add     rbx, 4
    add     r12, 4
    add     r14, 4
    dec     rdi
    jnz     .Lkloop_tail
    jmp     .Lepilogue

.p2align 5
.Lkloop_full:
    vmovups     ymm8, [rcx]
    add         rcx, r13
    vbroadcastss ymm9, [rsi]
    vfmadd231ps ymm0, ymm9, ymm8
    vbroadcastss ymm9, [r8]
    vfmadd231ps ymm1, ymm9, ymm8
    vbroadcastss ymm9, [r10]
    vfmadd231ps ymm2, ymm9, ymm8
    vbroadcastss ymm9, [r11]
    vfmadd231ps ymm3, ymm9, ymm8
    vbroadcastss ymm9, [rax]
    vfmadd231ps ymm4, ymm9, ymm8
    vbroadcastss ymm9, [rbx]
    vfmadd231ps ymm5, ymm9, ymm8
    vbroadcastss ymm9, [r12]
    vfmadd231ps ymm6, ymm9, ymm8
    vbroadcastss ymm9, [r14]
    vfmadd231ps ymm7, ymm9, ymm8
    add     rsi, 4
    add     r8,  4
    add     r10, 4
    add     r11, 4
    add     rax, 4
    add     rbx, 4
    add     r12, 4
    add     r14, 4
    dec     rdi
    jnz     .Lkloop_full

.Lepilogue:
    vmaskmovps [r9], ymm15, ymm0

    lea     rbp, [r9 + rdx]
    cmp     qword ptr [rsp + 64], 1
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm1
1:
    lea     rbp, [r9 + rdx*2]
    cmp     qword ptr [rsp + 64], 2
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm2
1:
    lea     rbp, [r9 + rdx*2]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 3
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm3
1:
    lea     rbp, [r9 + rdx*4]
    cmp     qword ptr [rsp + 64], 4
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm4
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 5
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm5
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 6
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm6
1:
    lea     rbp, [r9 + rdx*4]
    add     rbp, rdx
    add     rbp, rdx
    add     rbp, rdx
    cmp     qword ptr [rsp + 64], 7
    jbe     1f
    vmaskmovps [rbp], ymm15, ymm7
1:

    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

.size sgemm_avx2_micro, .-sgemm_avx2_micro

.section .rodata
.p2align 5
.Lcol_masks:
    .long 0, 0, 0, 0, 0, 0, 0, 0
    .long -1, 0, 0, 0, 0, 0, 0, 0
    .long -1, -1, 0, 0, 0, 0, 0, 0
    .long -1, -1, -1, 0, 0, 0, 0, 0
    .long -1, -1, -1, -1, 0, 0, 0, 0
    .long -1, -1, -1, -1, -1, 0, 0, 0
    .long -1, -1, -1, -1, -1, -1, 0, 0
    .long -1, -1, -1, -1, -1, -1, -1, 0
    .long -1, -1, -1, -1, -1, -1, -1, -1

.section .note.GNU-stack, "", @progbits
