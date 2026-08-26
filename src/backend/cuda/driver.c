#include "driver.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *sym;
    size_t offset;
} bind_entry;

static pg_cuda_drv g_drv;
static int g_state = -1;

#define BIND(field) { #field, offsetof(pg_cuda_drv, field) }

static const bind_entry g_binds[] = {
    BIND(init),
    BIND(device_get),
    BIND(ctx_create),
    BIND(mem_alloc),
    BIND(mem_free),
    BIND(memcpy_h2d),
    BIND(memcpy_d2h),
    BIND(ctx_sync),
    BIND(module_load_data),
    BIND(module_get_function),
    BIND(launch_kernel),
};

const pg_cuda_drv *pg_cuda_drv_get(pg_status *err)
{
    if (g_state == 0)
        return &g_drv;
    if (g_state == 1) {
        if (err)
            *err = PG_ERR_UNSUPPORTED;
        return NULL;
    }

    memset(&g_drv, 0, sizeof(g_drv));
    g_drv.handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_drv.handle)
        goto fail;

    for (size_t i = 0; i < sizeof(g_binds) / sizeof(g_binds[0]); i++) {
        void *sym = dlsym(g_drv.handle, g_binds[i].sym);
        if (!sym)
            goto fail;
        *(void **)((char *)&g_drv + g_binds[i].offset) = sym;
    }

    if (g_drv.init(0) != 0)
        goto fail_unloaded;

    g_state = 0;
    return &g_drv;

fail_unloaded:
    dlclose(g_drv.handle);
    memset(&g_drv, 0, sizeof(g_drv));
fail:
    g_state = 1;
    if (err)
        *err = PG_ERR_UNSUPPORTED;
    return NULL;
}
