#include "conv.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../backend/cpu/gemm.h"
#include "../thread/pool.h"

// im2col + GEMM based conv, chunked to limit memory
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

    size_t K = Cin * kh * kw;
    size_t Ncols = OH * OW;
    // heuristic: for very small problems naive may be faster due to im2col overhead
    // but GEMM path is still fast for small; we keep GEMM.
    // Chunk size to limit col buffer: 1024 columns or Ncols
    const size_t CHUNK = 1024;
    // For tiny Ncols (<256) just do one chunk
    // Allocate col buffer for max chunk
    size_t max_chunk = Ncols < CHUNK ? Ncols : CHUNK;
    // quick path for small total: if K*Ncols < 65536, single full col may be smaller than chunk loops overhead
    // we keep chunked anyway to simplify.

    // For each batch
    for (size_t n = 0; n < N; ++n) {
        const float *x_batch = x->data + n * Cin * H * W;
        float *out_batch = out->data + n * Cout * OH * OW;

        // process output columns in chunks
        for (size_t col_base = 0; col_base < Ncols; col_base += max_chunk) {
            size_t cur_ncols = Ncols - col_base;
            if (cur_ncols > max_chunk) cur_ncols = max_chunk;

            float *col = (float*)malloc(K * cur_ncols * sizeof(float));
            if (!col) {
                // fallback to naive for this batch/chunk if alloc fails
                for (size_t co = 0; co < Cout; ++co) {
                    for (size_t chunk_j = 0; chunk_j < cur_ncols; ++chunk_j) {
                        size_t gcol = col_base + chunk_j;
                        size_t oh = gcol / OW;
                        size_t ow = gcol % OW;
                        float acc = b ? b->data[co] : 0.0f;
                        // will be overwritten by GEMM fallback naive per element
                        // compute directly
                        for (size_t ci = 0; ci < Cin; ++ci) {
                            for (size_t ki = 0; ki < kh; ++ki) {
                                for (size_t kj = 0; kj < kw; ++kj) {
                                    long ih = (long)oh * s - p + (long)ki;
                                    long iw = (long)ow * s - p + (long)kj;
                                    if (ih >= 0 && ih < (long)H && iw >= 0 && iw < (long)W) {
                                        size_t x_idx = (ci * H + (size_t)ih) * W + (size_t)iw;
                                        size_t w_idx = ((co * Cin + ci) * kh + ki) * kw + kj;
                                        acc += x_batch[x_idx] * w->data[w_idx];
                                    }
                                }
                            }
                        }
                        out_batch[co * Ncols + gcol] = acc;
                    }
                }
                continue;
            }

            // fill col: K rows, cur_ncols cols, row-major K x cur_ncols
            // col[row*cur_ncols + col_idx] = img patch
            for (size_t chunk_j = 0; chunk_j < cur_ncols; ++chunk_j) {
                size_t gcol = col_base + chunk_j;
                size_t oh = gcol / OW;
                size_t ow = gcol % OW;
                // for each cin, kh, kw
                for (size_t ci = 0; ci < Cin; ++ci) {
                    for (size_t ki = 0; ki < kh; ++ki) {
                        for (size_t kj = 0; kj < kw; ++kj) {
                            size_t row = (ci * kh + ki) * kw + kj;
                            long ih = (long)oh * s - p + (long)ki;
                            long iw = (long)ow * s - p + (long)kj;
                            float v = 0.0f;
                            if (ih >= 0 && ih < (long)H && iw >= 0 && iw < (long)W) {
                                size_t x_idx = (ci * H + (size_t)ih) * W + (size_t)iw;
                                v = x_batch[x_idx];
                            }
                            col[row * cur_ncols + chunk_j] = v;
                        }
                    }
                }
            }

            // GEMM: Cout x cur_ncols = (Cout x K) * (K x cur_ncols)
            // w is Cout x K, lda=K
            // col is K x cur_ncols, ldb=cur_ncols
            // out chunk is Cout x Ncols with ldc=Ncols, pointer at col_base offset
            // out already zeroed, GEMM adds, but our GEMM zeros C first, so we need to handle bias after
            // GEMM will zero the chunk region (memset each row's cur_ncols segment correctly)
            // However pg_cpu_gemm zeros full rows (n=cur_ncols) but with ldc=Ncols and pointer offset,
            // it will zero exactly the chunk segment (since it zeros via memset per row with n* sizeof(float) at correct offset)
            // That is correct for our chunked pointer.
            float *c_ptr = out_batch + col_base; // row 0 offset, GEMM will handle per-row ldc
            pg_cpu_gemm(Cout, cur_ncols, K, w->data, K, col, cur_ncols, c_ptr, Ncols);

            // add bias if present (GEMM result currently is w*col, bias needs to be added)
            if (b) {
                for (size_t co = 0; co < Cout; ++co) {
                    float bv = b->data[co];
                    float *row_ptr = out_batch + co * Ncols + col_base;
                    for (size_t j = 0; j < cur_ncols; ++j) row_ptr[j] += bv;
                }
            }

            free(col);
        }
    }
    return out;
}
