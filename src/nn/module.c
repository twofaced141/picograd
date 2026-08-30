#include "module.h"
#include "../opt/sgd.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct pg_module {
    pg_node **params;
    size_t n, cap;
};

pg_module *pg_module_new(void)
{
    pg_module *m = malloc(sizeof(*m));
    if (!m)
        return NULL;
    m->params = NULL;
    m->n = 0;
    m->cap = 0;
    return m;
}

void pg_module_free(pg_module *m)
{
    if (!m)
        return;
    for (size_t i = 0; i < m->n; i++)
        pg_node_free(m->params[i]);
    free(m->params);
    free(m);
}

void pg_module_register(pg_module *m, pg_node *param)
{
    assert(m && param);
    if (m->n == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 8;
        pg_node **np = realloc(m->params, nc * sizeof(*np));
        if (!np) {
            assert(np);
            return;
        }
        m->params = np;
        m->cap = nc;
    }
    m->params[m->n++] = pg_node_retain(param);
}

size_t pg_module_num_params(const pg_module *m) { return m ? m->n : 0; }

pg_node **pg_module_params(pg_module *m) { return m ? m->params : NULL; }

int pg_module_add_to_adam(pg_module *m, pg_adam *opt)
{
    assert(m && opt);
    for (size_t i = 0; i < m->n; i++)
        if (pg_adam_add_param(opt, m->params[i]) != 0)
            return -1;
    return 0;
}

int pg_module_add_to_adam_lr(pg_module *m, pg_adam *opt, float lr, float wd)
{
    assert(m && opt);
    for (size_t i = 0; i < m->n; i++)
        if (pg_adam_add_param_lr(opt, m->params[i], lr, wd) != 0)
            return -1;
    return 0;
}

int pg_module_add_to_sgd(pg_module *m, pg_sgd *opt){
    assert(m && opt);
    for(size_t i=0;i<m->n;i++) if(pg_sgd_add_param(opt, m->params[i])!=0) return -1;
    return 0;
}

void pg_module_zero_grad(pg_module *m){
    if(!m) return;
    for(size_t i=0;i<m->n;i++){
        pg_node *p=m->params[i];
        if(p->grad){ pg_tensor_free(p->grad); p->grad=NULL; }
    }
}

#define PG_MODULE_MAGIC "PGMD"
#define PG_MODULE_VERSION 1

bool pg_module_save_fp(const pg_module *m, FILE *fp){
    if(!m || !fp) return false;
    if(fwrite(PG_MODULE_MAGIC, 1, 4, fp) != 4) return false;
    uint32_t ver = PG_MODULE_VERSION;
    if(fwrite(&ver, 4, 1, fp) != 1) return false;
    uint64_t n = (uint64_t)m->n;
    if(fwrite(&n, 8, 1, fp) != 1) return false;
    for(size_t i=0;i<m->n;i++){
        pg_tensor *v = m->params[i]->value;
        if(!v) return false;
        if(!pg_tensor_save_fp(v, fp)) return false;
    }
    return true;
}

bool pg_module_save(const pg_module *m, const char *path){
    if(!path) return false;
    FILE *fp=fopen(path,"wb");
    if(!fp) return false;
    bool ok=pg_module_save_fp(m,fp);
    fclose(fp);
    return ok;
}

pg_module *pg_module_load_fp(FILE *fp){
    if(!fp) return NULL;
    char magic[4];
    if(fread(magic,1,4,fp)!=4) return NULL;
    if(memcmp(magic, PG_MODULE_MAGIC,4)!=0) return NULL;
    uint32_t ver=0;
    if(fread(&ver,4,1,fp)!=1) return NULL;
    if(ver!=PG_MODULE_VERSION) return NULL;
    uint64_t n=0;
    if(fread(&n,8,1,fp)!=1) return NULL;
    if(n>100000) return NULL;
    pg_module *m=pg_module_new();
    if(!m) return NULL;
    for(uint64_t i=0;i<n;i++){
        pg_tensor *t=pg_tensor_load_fp(fp);
        if(!t){ pg_module_free(m); return NULL; }
        pg_node *node = pg_var_from_tensor(t, true);
        pg_tensor_free(t);
        if(!node){ pg_module_free(m); return NULL; }
        // steal retain: pg_var_from_tensor already creates node with refs=1, register retains
        pg_module_register(m, node);
        pg_node_free(node); // drop extra ref, module holds one
    }
    return m;
}

pg_module *pg_module_load(const char *path){
    if(!path) return NULL;
    FILE *fp=fopen(path,"rb");
    if(!fp) return NULL;
    pg_module *m=pg_module_load_fp(fp);
    fclose(fp);
    return m;
}

bool pg_module_load_into_fp(pg_module *m, FILE *fp){
    if(!m || !fp) return false;
    char magic[4];
    if(fread(magic,1,4,fp)!=4) return false;
    if(memcmp(magic, PG_MODULE_MAGIC,4)!=0) return false;
    uint32_t ver=0;
    if(fread(&ver,4,1,fp)!=1) return false;
    if(ver!=PG_MODULE_VERSION) return false;
    uint64_t n=0;
    if(fread(&n,8,1,fp)!=1) return false;
    if(n != m->n) return false;
    for(uint64_t i=0;i<n;i++){
        pg_tensor *t=pg_tensor_load_fp(fp);
        if(!t) return false;
        pg_tensor *dst = m->params[i]->value;
        if(!dst || dst->numel != t->numel || dst->dtype != t->dtype || dst->ndim != t->ndim){
            pg_tensor_free(t);
            return false;
        }
        bool shape_ok=true;
        for(size_t d=0;d<dst->ndim;d++) if(dst->shape[d]!=t->shape[d]) shape_ok=false;
        if(!shape_ok){ pg_tensor_free(t); return false; }
        memcpy(dst->data_raw, t->data_raw, t->numel * t->elem_size);
        pg_tensor_free(t);
    }
    return true;
}

bool pg_module_load_into(pg_module *m, const char *path){
    if(!m || !path) return false;
    FILE *fp=fopen(path,"rb");
    if(!fp) return false;
    bool ok=pg_module_load_into_fp(m,fp);
    fclose(fp);
    return ok;
}
