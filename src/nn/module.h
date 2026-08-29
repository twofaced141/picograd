#ifndef PICOGRAD_NN_MODULE_H
#define PICOGRAD_NN_MODULE_H

#include "../autograd/autograd.h"
#include "../opt/adam.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A minimal parameter registry. A model collects its parameter nodes once,
 * then a single call registers them all with an optimizer (no manual wiring
 * of every weight/bias). Param nodes are retained until the module is freed. */
typedef struct pg_module pg_module;

pg_module *pg_module_new(void);
void pg_module_free(pg_module *m);

void pg_module_register(pg_module *m, pg_node *param);
size_t pg_module_num_params(const pg_module *m);
pg_node **pg_module_params(pg_module *m);

/* Add every registered parameter to the optimizer (shared cfg / lr). */
int pg_module_add_to_adam(pg_module *m, pg_adam *opt);
/* Add every registered parameter to the optimizer with a fixed per-group
 * lr / wd override (0 / <0 keeps the optimizer default). */
int pg_module_add_to_adam_lr(pg_module *m, pg_adam *opt, float lr, float wd);

#ifdef __cplusplus
}
#endif

#endif
