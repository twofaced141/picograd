#include <stddef.h>

#if defined(__ARM_FEATURE_SVE) || defined(PG_HAVE_SVE)
#include <arm_sve.h>

// SVE 8x16 microkernel, VL-agnostic
// n = up to 16 (for VL=256). For other VL, handler loops with predicates.
// m = up to 8, k arbitrary.
void sgemm_sve_micro(size_t k,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     float *c, size_t ldc,
                     size_t m, size_t n)
{
    if (m==0 || n==0) return;
    if (k==0) return;
    if (n > 16 || m > 8) {
        for (size_t i=0;i<m;i++){
            for (size_t j=0;j<n;j++){
                float acc=c[i*ldc+j];
                for(size_t p=0;p<k;p++) acc += a[i*lda+p]*b[p*ldb+j];
                c[i*ldc+j]=acc;
            }
        }
        return;
    }

    for (size_t j = 0; j < n; ) {
        svbool_t pg = svwhilelt_b32((unsigned)j, (unsigned)n);
        svfloat32_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        // load accumulators (only for active rows)
        if (m>0) acc0 = svld1_f32(pg, c + 0*ldc + j);
        if (m>1) acc1 = svld1_f32(pg, c + 1*ldc + j);
        if (m>2) acc2 = svld1_f32(pg, c + 2*ldc + j);
        if (m>3) acc3 = svld1_f32(pg, c + 3*ldc + j);
        if (m>4) acc4 = svld1_f32(pg, c + 4*ldc + j);
        if (m>5) acc5 = svld1_f32(pg, c + 5*ldc + j);
        if (m>6) acc6 = svld1_f32(pg, c + 6*ldc + j);
        if (m>7) acc7 = svld1_f32(pg, c + 7*ldc + j);

        for (size_t p = 0; p < k; p++) {
            svfloat32_t bv = svld1_f32(pg, b + p*ldb + j);
            if (m>0) {
                svfloat32_t av = svdup_n_f32(a[0*lda + p]);
                acc0 = svmla_f32_z(pg, acc0, bv, av);
            }
            if (m>1) {
                svfloat32_t av = svdup_n_f32(a[1*lda + p]);
                acc1 = svmla_f32_z(pg, acc1, bv, av);
            }
            if (m>2) {
                svfloat32_t av = svdup_n_f32(a[2*lda + p]);
                acc2 = svmla_f32_z(pg, acc2, bv, av);
            }
            if (m>3) {
                svfloat32_t av = svdup_n_f32(a[3*lda + p]);
                acc3 = svmla_f32_z(pg, acc3, bv, av);
            }
            if (m>4) {
                svfloat32_t av = svdup_n_f32(a[4*lda + p]);
                acc4 = svmla_f32_z(pg, acc4, bv, av);
            }
            if (m>5) {
                svfloat32_t av = svdup_n_f32(a[5*lda + p]);
                acc5 = svmla_f32_z(pg, acc5, bv, av);
            }
            if (m>6) {
                svfloat32_t av = svdup_n_f32(a[6*lda + p]);
                acc6 = svmla_f32_z(pg, acc6, bv, av);
            }
            if (m>7) {
                svfloat32_t av = svdup_n_f32(a[7*lda + p]);
                acc7 = svmla_f32_z(pg, acc7, bv, av);
            }
        }
        if (m>0) svst1_f32(pg, c + 0*ldc + j, acc0);
        if (m>1) svst1_f32(pg, c + 1*ldc + j, acc1);
        if (m>2) svst1_f32(pg, c + 2*ldc + j, acc2);
        if (m>3) svst1_f32(pg, c + 3*ldc + j, acc3);
        if (m>4) svst1_f32(pg, c + 4*ldc + j, acc4);
        if (m>5) svst1_f32(pg, c + 5*ldc + j, acc5);
        if (m>6) svst1_f32(pg, c + 6*ldc + j, acc6);
        if (m>7) svst1_f32(pg, c + 7*ldc + j, acc7);

        size_t vl = svcntw();
        j += vl;
        if (vl==0) break;
    }
}

#else
void sgemm_sve_micro(size_t k,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     float *c, size_t ldc,
                     size_t m, size_t n)
{
    if (m==0 || n==0) return;
    if (k==0) return;
    for (size_t i=0;i<m;i++){
        for (size_t j=0;j<n;j++){
            float acc=c[i*ldc+j];
            for(size_t p=0;p<k;p++) acc += a[i*lda+p]*b[p*ldb+j];
            c[i*ldc+j]=acc;
        }
    }
}
#endif

// SVE FP16/BF16: f16 via svcvt_f32_f16, bf16 via bfmmla/bfdot if SVE2-BF16
#if defined(__ARM_FEATURE_SVE) || defined(PG_HAVE_SVE)
#include "../../core/convert.h"
void hgemm_sve_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    if (m==0||n==0||k==0) return;
    if (m > 8 || n > 16) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float acc = c[i * ldc + j];
                for (size_t p = 0; p < k; p++)
                    acc += pg_f16_to_f32_scalar(a[i * lda + p]) * pg_f16_to_f32_scalar(b[p * ldb + j]);
                c[i * ldc + j] = acc;
            }
        }
        return;
    }
    for(size_t j=0; j<n; ){
        svbool_t pg = svwhilelt_b32((unsigned)j, (unsigned)n);
        svfloat32_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        if(m>0) acc0=svld1_f32(pg, c + 0*ldc + j);
        if(m>1) acc1=svld1_f32(pg, c + 1*ldc + j);
        if(m>2) acc2=svld1_f32(pg, c + 2*ldc + j);
        if(m>3) acc3=svld1_f32(pg, c + 3*ldc + j);
        if(m>4) acc4=svld1_f32(pg, c + 4*ldc + j);
        if(m>5) acc5=svld1_f32(pg, c + 5*ldc + j);
        if(m>6) acc6=svld1_f32(pg, c + 6*ldc + j);
        if(m>7) acc7=svld1_f32(pg, c + 7*ldc + j);
        for(size_t p=0;p<k;p++){
            svfloat16_t bh = svld1_f16(pg, (const float16_t*)(b + p*ldb + j));
            svfloat32_t bf = svcvt_f32_f16_x(pg, bh);
            if(m>0){ float av=pg_f16_to_f32_scalar(a[0*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc0=svmla_f32_x(pg, acc0, bf, va); }
            if(m>1){ float av=pg_f16_to_f32_scalar(a[1*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc1=svmla_f32_x(pg, acc1, bf, va); }
            if(m>2){ float av=pg_f16_to_f32_scalar(a[2*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc2=svmla_f32_x(pg, acc2, bf, va); }
            if(m>3){ float av=pg_f16_to_f32_scalar(a[3*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc3=svmla_f32_x(pg, acc3, bf, va); }
            if(m>4){ float av=pg_f16_to_f32_scalar(a[4*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc4=svmla_f32_x(pg, acc4, bf, va); }
            if(m>5){ float av=pg_f16_to_f32_scalar(a[5*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc5=svmla_f32_x(pg, acc5, bf, va); }
            if(m>6){ float av=pg_f16_to_f32_scalar(a[6*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc6=svmla_f32_x(pg, acc6, bf, va); }
            if(m>7){ float av=pg_f16_to_f32_scalar(a[7*lda+p]);
                svfloat32_t va=svdup_n_f32(av); acc7=svmla_f32_x(pg, acc7, bf, va); }
        }
        if(m>0) svst1_f32(pg, c + 0*ldc + j, acc0);
        if(m>1) svst1_f32(pg, c + 1*ldc + j, acc1);
        if(m>2) svst1_f32(pg, c + 2*ldc + j, acc2);
        if(m>3) svst1_f32(pg, c + 3*ldc + j, acc3);
        if(m>4) svst1_f32(pg, c + 4*ldc + j, acc4);
        if(m>5) svst1_f32(pg, c + 5*ldc + j, acc5);
        if(m>6) svst1_f32(pg, c + 6*ldc + j, acc6);
        if(m>7) svst1_f32(pg, c + 7*ldc + j, acc7);
        size_t vl=svcntw();
        j+=vl; if(vl==0) break;
    }
}
void bgemm_sve_micro(size_t k, const uint16_t *a, size_t lda, const uint16_t *b, size_t ldb, float *c, size_t ldc, size_t m, size_t n){
    if (m==0||n==0||k==0) return;
#if defined(__ARM_FEATURE_SVE_BF16)
    if(m<=8 && n<=16){
        for(size_t j=0; j<n; ){
            svbool_t pg = svwhilelt_b32((unsigned)j, (unsigned)n);
            svfloat32_t acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7;
            if(m>0) acc0=svld1_f32(pg, c+0*ldc+j);
            if(m>1) acc1=svld1_f32(pg, c+1*ldc+j);
            if(m>2) acc2=svld1_f32(pg, c+2*ldc+j);
            if(m>3) acc3=svld1_f32(pg, c+3*ldc+j);
            if(m>4) acc4=svld1_f32(pg, c+4*ldc+j);
            if(m>5) acc5=svld1_f32(pg, c+5*ldc+j);
            if(m>6) acc6=svld1_f32(pg, c+6*ldc+j);
            if(m>7) acc7=svld1_f32(pg, c+7*ldc+j);
            for(size_t p=0;p<k;p++){
                svbfloat16_t bh = svld1_bf16(pg, (const bfloat16_t*)(b + p*ldb + j));
                svfloat32_t bf = svcvt_f32_bf16_x(pg, bh);
                if(m>0){ float av=pg_bf16_to_f32_scalar(a[0*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc0=svmla_f32_x(pg, acc0, bf, va); }
                if(m>1){ float av=pg_bf16_to_f32_scalar(a[1*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc1=svmla_f32_x(pg, acc1, bf, va); }
                if(m>2){ float av=pg_bf16_to_f32_scalar(a[2*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc2=svmla_f32_x(pg, acc2, bf, va); }
                if(m>3){ float av=pg_bf16_to_f32_scalar(a[3*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc3=svmla_f32_x(pg, acc3, bf, va); }
                if(m>4){ float av=pg_bf16_to_f32_scalar(a[4*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc4=svmla_f32_x(pg, acc4, bf, va); }
                if(m>5){ float av=pg_bf16_to_f32_scalar(a[5*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc5=svmla_f32_x(pg, acc5, bf, va); }
                if(m>6){ float av=pg_bf16_to_f32_scalar(a[6*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc6=svmla_f32_x(pg, acc6, bf, va); }
                if(m>7){ float av=pg_bf16_to_f32_scalar(a[7*lda+p]);
                    svfloat32_t va=svdup_n_f32(av); acc7=svmla_f32_x(pg, acc7, bf, va); }
            }
            if(m>0) svst1_f32(pg, c+0*ldc+j, acc0);
            if(m>1) svst1_f32(pg, c+1*ldc+j, acc1);
            if(m>2) svst1_f32(pg, c+2*ldc+j, acc2);
            if(m>3) svst1_f32(pg, c+3*ldc+j, acc3);
            if(m>4) svst1_f32(pg, c+4*ldc+j, acc4);
            if(m>5) svst1_f32(pg, c+5*ldc+j, acc5);
            if(m>6) svst1_f32(pg, c+6*ldc+j, acc6);
            if(m>7) svst1_f32(pg, c+7*ldc+j, acc7);
            size_t vl=svcntw();
            j+=vl; if(vl==0) break;
        }
        return;
    }
#endif
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = c[i * ldc + j];
            for (size_t p = 0; p < k; p++)
                acc += pg_bf16_to_f32_scalar(a[i * lda + p]) * pg_bf16_to_f32_scalar(b[p * ldb + j]);
            c[i * ldc + j] = acc;
        }
    }
}
#endif
