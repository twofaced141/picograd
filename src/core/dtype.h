#ifndef PICOGRAD_CORE_DTYPE_H
#define PICOGRAD_CORE_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PG_DTYPE_F32  = 0,
    PG_DTYPE_F16  = 1,
    PG_DTYPE_BF16 = 2,
    PG_DTYPE_COUNT = 3
} pg_dtype;

// aliases for plan spec
#define PG_F32  PG_DTYPE_F32
#define PG_F16  PG_DTYPE_F16
#define PG_BF16 PG_DTYPE_BF16

static inline size_t pg_dtype_size(pg_dtype dt) {
    switch (dt) {
        case PG_DTYPE_F32:  return 4;
        case PG_DTYPE_F16:  return 2;
        case PG_DTYPE_BF16: return 2;
        default: return 0;
    }
}

static inline const char *pg_dtype_name(pg_dtype dt) {
    switch (dt) {
        case PG_DTYPE_F32:  return "f32";
        case PG_DTYPE_F16:  return "f16";
        case PG_DTYPE_BF16: return "bf16";
        default: return "unknown";
    }
}

static inline int pg_dtype_is_float(pg_dtype dt){
    return dt==PG_DTYPE_F32 || dt==PG_DTYPE_F16 || dt==PG_DTYPE_BF16;
}

#ifdef __cplusplus
}
#endif

#endif
