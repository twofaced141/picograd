#ifndef PICOGRAD_NN_MODULE_H
#define PICOGRAD_NN_MODULE_H

#include "../autograd/autograd.h"
#include "../opt/adam.h"
#include <stdbool.h>
#include <stdio.h>

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

// SGD support
struct pg_sgd;
int pg_module_add_to_sgd(pg_module *m, struct pg_sgd *opt);

// Zero grad for all params
void pg_module_zero_grad(pg_module *m);

// Checkpoint: save/load all param tensors (binary, order preserved)
// Format: uint64 nparams + for each: pg_tensor_save_fp
bool pg_module_save(const pg_module *m, const char *path);
bool pg_module_save_fp(const pg_module *m, FILE *fp);
pg_module *pg_module_load(const char *path);
pg_module *pg_module_load_fp(FILE *fp);
// Load values into existing module (shapes must match, nparams must match)
bool pg_module_load_into(pg_module *m, const char *path);
bool pg_module_load_into_fp(pg_module *m, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
