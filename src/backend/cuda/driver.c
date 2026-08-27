#include "driver.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *sym;
    size_t offset;
} bind_entry;

static pg_cuda_drv g_drv;
static int g_state = -1;

static const bind_entry g_binds[] = {
    { "cuInit", offsetof(pg_cuda_drv, init) },
    { "cuDeviceGet", offsetof(pg_cuda_drv, device_get) },
    { "cuCtxCreate", offsetof(pg_cuda_drv, ctx_create) },
    { "cuMemAlloc", offsetof(pg_cuda_drv, mem_alloc) },
    { "cuMemFree", offsetof(pg_cuda_drv, mem_free) },
    { "cuMemcpyHtoD", offsetof(pg_cuda_drv, memcpy_h2d) },
    { "cuMemcpyDtoH", offsetof(pg_cuda_drv, memcpy_d2h) },
    { "cuMemcpyDtoD", offsetof(pg_cuda_drv, memcpy_d2d) },
    { "cuMemsetD32", offsetof(pg_cuda_drv, memset_u32) },
    { "cuCtxSynchronize", offsetof(pg_cuda_drv, ctx_sync) },
    { "cuModuleLoadData", offsetof(pg_cuda_drv, module_load_data) },
    { "cuModuleGetFunction", offsetof(pg_cuda_drv, module_get_function) },
    { "cuLaunchKernel", offsetof(pg_cuda_drv, launch_kernel) },
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

    int debug = getenv("PG_CUDA_DEBUG") != NULL;
    memset(&g_drv, 0, sizeof(g_drv));
    const char *candidates[] = {
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so.1",
    };
    for (size_t ci = 0; ci < sizeof(candidates)/sizeof(candidates[0]); ci++) {
        g_drv.handle = dlopen(candidates[ci], RTLD_NOW | RTLD_LOCAL);
        if (g_drv.handle) {
            if (debug) fprintf(stderr, "picograd/cuda: dlopen ok %s\n", candidates[ci]);
            break;
        }
        if (debug) fprintf(stderr, "picograd/cuda: dlopen %s: %s\n", candidates[ci], dlerror());
    }
    if (!g_drv.handle) {
        goto fail;
    }

    for (size_t i = 0; i < sizeof(g_binds) / sizeof(g_binds[0]); i++) {
        void *sym = dlsym(g_drv.handle, g_binds[i].sym);
        if (!sym) {
            if (debug)
                fprintf(stderr, "picograd/cuda: missing %s\n",
                        g_binds[i].sym);
            goto fail_unloaded;
        }
        *(void **)((char *)&g_drv + g_binds[i].offset) = sym;
    }

    int rc = g_drv.init(0);
    if (rc != 0) {
        if (debug)
            fprintf(stderr, "picograd/cuda: cuInit -> %d\n", rc);
        goto fail_unloaded;
    }

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
