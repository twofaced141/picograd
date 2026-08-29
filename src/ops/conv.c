#include "conv.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

pg_tensor *pg_conv2d(const pg_tensor *x, const pg_tensor *w, const pg_tensor *b,
                     pg_conv2d_cfg cfg)
{
    if (!x || !w || !x->data || !w->data) return NULL;
    if (x->ndim != 4 || w->ndim != 4) return NULL;
    if (x->shape[1] != w->shape[1]) return NULL;
    if (b && (!b->data || b->ndim != 1 || b->shape[0] != w->shape[0])) return NULL;
    if (cfg.stride <= 0) return NULL;

    size_t N = x->shape[0];
    size_t Cin = x->shape[1];
    size_t H = x->shape[2];
    size_t W = x->shape[3];
    size_t Cout = w->shape[0];
    size_t kh = w->shape[2];
    size_t kw = w->shape[3];
    int s = cfg.stride, p = cfg.padding;

    size_t OH = pg_conv2d_out_size(H, kh, s, p);
    size_t OW = pg_conv2d_out_size(W, kw, s, p);
    if (OH == 0 || OW == 0)
        return NULL;

    size_t oshape[4] = {N, Cout, OH, OW};
    pg_tensor *out = pg_tensor_zeros(4, oshape);
    if (!out)
        return NULL;

    for (size_t n = 0; n < N; n++)
        for (size_t co = 0; co < Cout; co++)
            for (size_t oh = 0; oh < OH; oh++)
                for (size_t ow = 0; ow < OW; ow++) {
                    float acc = b ? b->data[co] : 0.0f;
                    for (size_t ci = 0; ci < Cin; ci++)
                        for (size_t i = 0; i < kh; i++)
                            for (size_t j = 0; j < kw; j++) {
                                long ph = (long)oh * s - p + (long)i;
                                long pw = (long)ow * s - p + (long)j;
                                if (ph >= 0 && ph < (long)H && pw >= 0 && pw < (long)W)
                                    acc += x->data[(n * Cin + ci) * H * W +
                                                   (size_t)ph * W + (size_t)pw] *
                                           w->data[((co * Cin + ci) * kh + i) * kw + j];
                            }
                    out->data[(n * Cout + co) * OH * OW + oh * OW + ow] = acc;
                }
    return out;
}
