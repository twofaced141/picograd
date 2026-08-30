#include "norm.h"
#include "../thread/pool.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const float *x; float *y; const float *w; const float *b; float eps; size_t N; } ln_par_t;

static void layernorm_par(void *ctx, size_t s, size_t e){
    ln_par_t *p=ctx;
    const float *x=p->x; float *y=p->y; const float *w=p->w; const float *b=p->b; float eps=p->eps; size_t N=p->N;
    for(size_t o=s;o<e;o++){
        const float *row=x + o*N;
        float *orow=y + o*N;
        double sum=0;
        for(size_t i=0;i<N;i++) sum+=row[i];
        float mean=(float)(sum / (double)N);
        double var=0;
        for(size_t i=0;i<N;i++){ double d=row[i]-mean; var+=d*d; }
        var/= (double)N;
        float rstd = 1.0f / sqrtf((float)var + eps);
        for(size_t i=0;i<N;i++){
            float v=(row[i]-mean)*rstd;
            if(w) v*=w[i];
            if(b) v+=b[i];
            orow[i]=v;
        }
    }
}

pg_tensor *pg_layernorm(const pg_tensor *x, const pg_tensor *weight, const pg_tensor *bias, float eps){
    if (!x || !x->data) return NULL;
    if (eps <= 0 || x->ndim < 1) return NULL;
    size_t N = x->shape[x->ndim-1];
    if(weight){ if(weight->ndim!=1 || weight->shape[0]!=N || !weight->data) return NULL; }
    if(bias){ if(bias->ndim!=1 || bias->shape[0]!=N || !bias->data) return NULL; }
    pg_tensor *y = pg_tensor_empty(x->ndim, x->shape);
    if(!y) return NULL;
    size_t outer = x->numel / N;
    if(outer >= 32 && outer * N >= 65536){
        ln_par_t ctx={x->data, y->data, weight?weight->data:NULL, bias?bias->data:NULL, eps, N};
        pg_parallel_for(outer, 8, layernorm_par, &ctx);
    } else {
        const float *w = weight?weight->data:NULL;
        const float *b = bias?bias->data:NULL;
        for(size_t o=0;o<outer;o++){
            const float *row=x->data + o*N;
            float *orow=y->data + o*N;
            double sum=0; for(size_t i=0;i<N;i++) sum+=row[i];
            float mean=(float)(sum/(double)N);
            double var=0; for(size_t i=0;i<N;i++){ double d=row[i]-mean; var+=d*d; }
            var/= (double)N;
            float rstd=1.0f/sqrtf((float)var + eps);
            for(size_t i=0;i<N;i++){
                float v=(row[i]-mean)*rstd;
                if(w) v*=w[i];
                if(b) v+=b[i];
                orow[i]=v;
            }
        }
    }
    return y;
}

typedef struct { const float *x; float *y; const float *w; float eps; size_t N; } rms_par_t;
static void rmsnorm_par(void *ctx, size_t s, size_t e){
    rms_par_t *p=ctx;
    for(size_t o=s;o<e;o++){
        const float *row=p->x + o*p->N;
        float *orow=p->y + o*p->N;
        double sumsq=0; for(size_t i=0;i<p->N;i++) sumsq+=(double)row[i]*row[i];
        float rstd=1.0f/sqrtf((float)(sumsq/p->N) + p->eps);
        for(size_t i=0;i<p->N;i++){ float v=row[i]*rstd; if(p->w) v*=p->w[i]; orow[i]=v; }
    }
}

pg_tensor *pg_rmsnorm(const pg_tensor *x, const pg_tensor *weight, float eps){
    if (!x || !x->data || x->ndim < 1) return NULL;
    if (eps <= 0) return NULL;
    size_t N=x->shape[x->ndim-1];
    if(weight){ if(weight->ndim!=1 || weight->shape[0]!=N || !weight->data) return NULL; }
    pg_tensor *y=pg_tensor_empty(x->ndim, x->shape);
    if(!y) return NULL;
    size_t outer=x->numel / N;
    if(outer >= 32 && outer * N >= 65536){
        rms_par_t ctx={x->data, y->data, weight?weight->data:NULL, eps, N};
        pg_parallel_for(outer, 8, rmsnorm_par, &ctx);
    } else {
        const float *w=weight?weight->data:NULL;
        for(size_t o=0;o<outer;o++){
            const float *row=x->data+o*N; float *orow=y->data+o*N;
            double sumsq=0; for(size_t i=0;i<N;i++) sumsq+=(double)row[i]*row[i];
            float rstd=1.0f/sqrtf((float)(sumsq/N)+eps);
            for(size_t i=0;i<N;i++){ float v=row[i]*rstd; if(w) v*=w[i]; orow[i]=v; }
        }
    }
    return y;
}

static bool __attribute__((unused)) bn_is_contiguous(const pg_tensor *t){
    size_t acc=1;
    for(size_t i=t->ndim;i-- >0;){
        if(t->stride[i]!=acc) return false;
        acc*=t->shape[i];
    }
    return true;
}

pg_tensor *pg_batchnorm2d(const pg_tensor *x, const pg_tensor *weight, const pg_tensor *bias,
                          pg_tensor *running_mean, pg_tensor *running_var,
                          float eps, float momentum, bool training){
    if(!x || !x->data) return NULL;
    if(x->ndim < 2) return NULL;
    if(x->dtype != PG_DTYPE_F32) return NULL;
    if(eps <= 0) return NULL;
    if(momentum < 0 || momentum > 1) return NULL;
    size_t C = x->shape[1];
    if(weight){ if(weight->ndim!=1 || weight->shape[0]!=C || !weight->data || weight->dtype!=PG_DTYPE_F32) return NULL; }
    if(bias){ if(bias->ndim!=1 || bias->shape[0]!=C || !bias->data || bias->dtype!=PG_DTYPE_F32) return NULL; }
    if(running_mean){ if(running_mean->ndim!=1 || running_mean->shape[0]!=C || !running_mean->data || running_mean->dtype!=PG_DTYPE_F32) return NULL; }
    if(running_var){ if(running_var->ndim!=1 || running_var->shape[0]!=C || !running_var->data || running_var->dtype!=PG_DTYPE_F32) return NULL; }
    if(!training && (!running_mean || !running_var)) return NULL;
    // For simplicity require contiguous (most cases are)
    // but we will handle non-contig via generic strided loop if needed
    pg_tensor *y = pg_tensor_empty(x->ndim, x->shape);
    if(!y) return NULL;

    size_t perChannel = x->numel / C;
    double *mean = (double*)calloc(C, sizeof(double));
    double *var = (double*)calloc(C, sizeof(double));
    float *invStd = (float*)calloc(C, sizeof(float));
    if(!mean || !var || !invStd){ free(mean); free(var); free(invStd); pg_tensor_free(y); return NULL; }

    if(training){
        // first pass: sum
        // Use strided iteration to correctly map channel
        size_t idx[PG_MAX_NDIM]={0};
        size_t off=0;
        // we need x->stride to compute off; start at 0
        for(size_t p=0; p<x->numel; p++){
            size_t c = idx[1];
            double v = (double)x->data[off];
            mean[c] += v;
            // advance idx/off
            for(size_t d=x->ndim; d-- >0;){
                idx[d]++;
                off += x->stride[d];
                if(idx[d] < x->shape[d]) break;
                idx[d]=0;
                off -= x->stride[d] * x->shape[d];
            }
        }
        for(size_t c=0;c<C;c++) mean[c] /= (double)perChannel;
        // second pass: var
        memset(idx,0,sizeof(idx));
        off=0;
        for(size_t p=0; p<x->numel; p++){
            size_t c = idx[1];
            double d = (double)x->data[off] - mean[c];
            var[c] += d*d;
            for(size_t d2=x->ndim; d2-- >0;){
                idx[d2]++;
                off += x->stride[d2];
                if(idx[d2] < x->shape[d2]) break;
                idx[d2]=0;
                off -= x->stride[d2] * x->shape[d2];
            }
        }
        for(size_t c=0;c<C;c++){
            var[c] /= (double)perChannel;
            invStd[c] = 1.0f / sqrtf((float)var[c] + eps);
        }
        // update running stats if provided
        if(running_mean && running_var){
            for(size_t c=0;c<C;c++){
                // PyTorch style: running = momentum*running + (1-momentum)*batch
                // Note: momentum here is as in pytorch: running = (1-momentum)*running + momentum*batch? Use provided semantics: running = momentum*running + (1-momentum)*batch
                running_mean->data[c] = momentum * running_mean->data[c] + (1.0f - momentum) * (float)mean[c];
                running_var->data[c] = momentum * running_var->data[c] + (1.0f - momentum) * (float)var[c];
            }
        }
    } else {
        // inference: use running stats
        for(size_t c=0;c<C;c++){
            mean[c] = (double)running_mean->data[c];
            var[c] = (double)running_var->data[c];
            invStd[c] = 1.0f / sqrtf((float)var[c] + eps);
        }
    }

    const float *w = weight ? weight->data : NULL;
    const float *b = bias ? bias->data : NULL;
    // third pass: y = gamma * (x - mean)*invStd + beta
    size_t idx[PG_MAX_NDIM]={0};
    size_t off_x=0, off_y=0;
    for(size_t p=0; p<x->numel; p++){
        size_t c = idx[1];
        float xv = x->data[off_x];
        float xhat = (xv - (float)mean[c]) * invStd[c];
        float yv = xhat;
        if(w) yv *= w[c];
        if(b) yv += b[c];
        y->data[off_y] = yv;
        for(size_t d=x->ndim; d-- >0;){
            idx[d]++;
            off_x += x->stride[d];
            off_y += y->stride[d];
            if(idx[d] < x->shape[d]) break;
            idx[d]=0;
            off_x -= x->stride[d] * x->shape[d];
            off_y -= y->stride[d] * y->shape[d];
        }
    }

    free(mean); free(var); free(invStd);
    return y;
}

pg_tensor *pg_batchnorm(const pg_tensor *x, const pg_tensor *weight, const pg_tensor *bias, float eps){
    return pg_batchnorm2d(x, weight, bias, NULL, NULL, eps, 0.1f, true);
}
