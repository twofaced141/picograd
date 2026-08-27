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
    assert(x && x->data);
    assert(eps > 0);
    size_t N = x->shape[x->ndim-1];
    if(weight){ assert(weight->ndim==1 && weight->shape[0]==N); assert(weight->data); }
    if(bias){ assert(bias->ndim==1 && bias->shape[0]==N); assert(bias->data); }
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
    assert(x && x->data);
    size_t N=x->shape[x->ndim-1];
    if(weight){ assert(weight->ndim==1 && weight->shape[0]==N); }
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
