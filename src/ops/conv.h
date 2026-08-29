#ifndef PICOGRAD_OPS_CONV_H
#define PICOGRAD_OPS_CONV_H

#include "../core/tensor.h"
#include <stdbool.h>

typedef struct {
    int stride;
    int padding;
} pg_conv2d_cfg;

static inline pg_conv2d_cfg pg_conv2d_cfg_default(void)
{
    return (pg_conv2d_cfg){1, 0};
}

/* Output spatial size for a valid 2D convolution:
 *   out = floor((in + 2*pad - k) / stride) + 1   (0 if the result would be <= 0) */
static inline size_t pg_conv2d_out_size(size_t in_size, size_t k, int stride, int pad)
{
    long v = (long)in_size + 2L * pad - (long)k;
    if (v < 0)
        return 0;
    return (size_t)(v / (long)stride) + 1;
}

/* NCHW 2D convolution.
 *   x : [N, Cin, H, W]
 *   w : [Cout, Cin, kh, kw]
 *   b : [Cout] or NULL
 * Returns [N, Cout, OH, OW]; NULL on error. */
pg_tensor *pg_conv2d(const pg_tensor *x, const pg_tensor *w, const pg_tensor *b,
                     pg_conv2d_cfg cfg);

#endif
