#ifndef PICOGRAD_BACKEND_H
#define PICOGRAD_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "../core/dtype.h"

/* Forward declaration for GPU exec helper */
struct pg_tensor;
typedef struct pg_tensor pg_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PG_DEV_CPU = 0,
    PG_DEV_CUDA,
    PG_DEV_METAL,
    PG_DEV_HIP,
    PG_DEV_ROCM = PG_DEV_HIP,
} pg_devtype;

typedef enum {
    PG_OK = 0,
    PG_ERR_UNSUPPORTED = -1,
    PG_ERR_ALLOC = -2,
    PG_ERR_COPY = -3,
    PG_ERR_SYNC = -4,
    PG_ERR_GEMM = -5,
} pg_status;

pg_status pg_set_device(pg_devtype dev);
pg_devtype pg_get_device(void);

#define PG_MAX_OP_NDIM 8

typedef enum {
    PG_MAP_EXP = 0,
    PG_MAP_LOG,
    PG_MAP_SIN,
    PG_MAP_COS,
    PG_MAP_SQRT,
    PG_MAP_NEG,
    PG_MAP_ABS,
    PG_MAP_ERF,
    PG_MAP_RELU,
    PG_MAP_SIGMOID,
    PG_MAP_TANH,
} pg_map_op;

typedef enum {
    PG_BIN_ADD = 0,
    PG_BIN_SUB,
    PG_BIN_MUL,
    PG_BIN_DIV,
    PG_BIN_SIG_BW,
    PG_BIN_TANH_BW,
    PG_BIN_RELU_BW,
} pg_bin_op;

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned shape[PG_MAX_OP_NDIM];
} pg_k_shape;

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned shape[PG_MAX_OP_NDIM];
    unsigned sa[PG_MAX_OP_NDIM];
    unsigned sb[PG_MAX_OP_NDIM];
} pg_k_bin_args;

typedef struct {
    unsigned ndim;
    unsigned numel;
    unsigned shape[PG_MAX_OP_NDIM];
    unsigned s[PG_MAX_OP_NDIM];
} pg_k_strides;

#define PG_MAX_NDIM_K PG_MAX_OP_NDIM

pg_status pg_op_fill(void *p, size_t nbytes, float v);
pg_status pg_op_copy_d2d(void *dst, const void *src, size_t nbytes);

pg_status pg_op_map(float *out, const float *src, size_t n, int op);
pg_status pg_op_bin(float *out, const float *a, const float *b,
                    size_t n, int op, const pg_k_bin_args *args);
pg_status pg_op_accum_scatter(float *dst, const float *src, float scale,
                              const pg_k_strides *args);
pg_status pg_op_sum_axis(float *out, const float *src, float scale,
                         size_t outer, size_t len, size_t inner,
                         size_t keepdim_stride);
pg_status pg_op_softmax(float *out, const float *src,
                        size_t outer, size_t len, size_t inner);
pg_status pg_op_copy_strided(float *dst, const float *src,
                             const pg_k_strides *args);

void *pg_dev_malloc(size_t nbytes);
void pg_dev_free(void *p);

pg_status pg_copy_h2d(void *dst, const void *src, size_t nbytes);
pg_status pg_copy_d2h(void *dst, const void *src, size_t nbytes);
pg_status pg_dev_sync(void);

/* GPU buffer helper - RAII-like wrapper for device buffers */
typedef struct {
    float *ptr;
    size_t nbytes;
} pg_dev_buf;

pg_dev_buf pg_dev_buf_new(size_t nbytes);
void pg_dev_buf_free(pg_dev_buf *buf);

void pg_gemm(size_t m, size_t n, size_t k,
             const float *a, size_t lda,
             const float *b, size_t ldb,
             float *c, size_t ldc);

// mixed-precision GEMM: A/B are dtype (F16/BF16 as uint16_t), C is f32 accum
void pg_gemm_ex(pg_dtype dtype,
                size_t m, size_t n, size_t k,
                const void *a, size_t lda,
                const void *b, size_t ldb,
                float *c, size_t ldc);
void pg_hgemm(size_t m, size_t n, size_t k,
              const uint16_t *a, size_t lda,
              const uint16_t *b, size_t ldb,
              float *c, size_t ldc);
void pg_bgemm(size_t m, size_t n, size_t k,
              const uint16_t *a, size_t lda,
              const uint16_t *b, size_t ldb,
              float *c, size_t ldc);

#ifdef __cplusplus
}
#endif

#endif
