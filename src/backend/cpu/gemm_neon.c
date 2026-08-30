#include <stddef.h>

#if defined(__aarch64__)
#include <arm_neon.h>

void sgemm_neon_micro(size_t k,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float *c, size_t ldc,
                      size_t m, size_t n)
{
    if (m==0 || n==0) return;
    if (k==0) return;
    // generic fallback for oversize n (>8) - scalar
    if (n > 8 || m > 8) {
        for (size_t i=0;i<m;i++){
            for (size_t j=0;j<n;j++){
                float acc=c[i*ldc+j];
                for(size_t p=0;p<k;p++) acc += a[i*lda+p] * b[p*ldb+j];
                c[i*ldc+j]=acc;
            }
        }
        return;
    }
    size_t n_vec = n / 4;
    size_t n_tail = n % 4;

    float32x4_t acc[8][2];
    float tail_acc[8][4] = {{0}};
    // init
    for (size_t i=0;i<m;i++){
        float *crow = c + i*ldc;
        if (n_vec >= 1) acc[i][0] = vld1q_f32(crow);
        else acc[i][0] = vdupq_n_f32(0);
        if (n_vec >= 2) acc[i][1] = vld1q_f32(crow+4);
        else acc[i][1] = vdupq_n_f32(0);
        for (size_t t=0; t<n_tail; t++) tail_acc[i][t] = crow[n_vec*4 + t];
    }
    for (size_t p=0; p<k; p++){
        const float *brow = b + p*ldb;
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
        if (n_vec >= 1) b0 = vld1q_f32(brow);
        if (n_vec >= 2) b1 = vld1q_f32(brow+4);
        for (size_t i=0;i<m;i++){
            float av = a[i*lda + p];
            float32x4_t va = vdupq_n_f32(av);
            if (n_vec >= 1) acc[i][0] = vmlaq_f32(acc[i][0], va, b0);
            if (n_vec >= 2) acc[i][1] = vmlaq_f32(acc[i][1], va, b1);
            for (size_t t=0; t<n_tail; t++){
                tail_acc[i][t] += av * brow[n_vec*4 + t];
            }
        }
    }
    for (size_t i=0;i<m;i++){
        float *crow = c + i*ldc;
        if (n_vec >= 1) vst1q_f32(crow, acc[i][0]);
        if (n_vec >= 2) vst1q_f32(crow+4, acc[i][1]);
        for (size_t t=0; t<n_tail; t++) crow[n_vec*4 + t] = tail_acc[i][t];
    }
}

#else
// fallback stub for non-aarch64 builds (should not be linked, but keep symbol)
void sgemm_neon_micro(size_t k,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float *c, size_t ldc,
                      size_t m, size_t n)
{
    // generic fallback
    if (m==0 || n==0) return;
    if (k==0) return;
    for (size_t i=0;i<m;i++){
        for (size_t j=0;j<n;j++){
            float acc=c[i*ldc+j];
            for(size_t p=0;p<k;p++) acc += a[i*lda+p] * b[p*ldb+j];
            c[i*ldc+j]=acc;
        }
    }
}
#endif
