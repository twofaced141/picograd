#include "module.h"

#include <assert.h>
#include <stdlib.h>

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
        assert(np);
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
