#include "driver.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *sym;
    size_t offset;
} bind_entry;

static pg_hip_drv g_drv;
static int g_state = -1; /* -1 unknown, 0 ok, 1 failed */

static const bind_entry g_hip_binds[] = {
    {"hipGetDeviceCount",      offsetof(pg_hip_drv, getDeviceCount)},
    {"hipGetDevice",           offsetof(pg_hip_drv, getDevice)},
    {"hipSetDevice",           offsetof(pg_hip_drv, setDevice)},
    {"hipMalloc",              offsetof(pg_hip_drv, malloc)},
    {"hipFree",                offsetof(pg_hip_drv, free)},
    {"hipMemcpy",              offsetof(pg_hip_drv, memcpy)},
    {"hipMemset",              offsetof(pg_hip_drv, memset)},
    {"hipDeviceSynchronize",   offsetof(pg_hip_drv, deviceSynchronize)},
    {"hipModuleLoadData",      offsetof(pg_hip_drv, moduleLoadData)},
    {"hipModuleGetFunction",   offsetof(pg_hip_drv, moduleGetFunction)},
    {"hipModuleLaunchKernel",  offsetof(pg_hip_drv, moduleLaunchKernel)},
};

static const bind_entry g_rtc_binds[] = {
    {"hiprtcCreateProgram",    offsetof(pg_hip_drv, rtcCreateProgram)},
    {"hiprtcCompileProgram",   offsetof(pg_hip_drv, rtcCompileProgram)},
    {"hiprtcGetCodeSize",      offsetof(pg_hip_drv, rtcGetCodeSize)},
    {"hiprtcGetCode",          offsetof(pg_hip_drv, rtcGetCode)},
    {"hiprtcDestroyProgram",   offsetof(pg_hip_drv, rtcDestroyProgram)},
    {"hiprtcGetProgramLogSize",offsetof(pg_hip_drv, rtcGetProgramLogSize)},
    {"hiprtcGetProgramLog",    offsetof(pg_hip_drv, rtcGetProgramLog)},
};

const pg_hip_drv *pg_hip_drv_get(pg_status *err)
{
    if (g_state == 0)
        return &g_drv;
    if (g_state == 1) {
        if (err) *err = PG_ERR_UNSUPPORTED;
        return NULL;
    }

    int debug = getenv("PG_HIP_DEBUG") != NULL;
    memset(&g_drv, 0, sizeof(g_drv));

    const char *hip_candidates[] = {
        "libamdhip64.so",
        "libamdhip64.so.5",
        "libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.5",
        "/opt/rocm/lib/libamdhip64.so.6",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        "/usr/lib/libamdhip64.so",
    };

    for (size_t ci = 0; ci < sizeof(hip_candidates)/sizeof(hip_candidates[0]); ci++) {
        g_drv.hip_handle = dlopen(hip_candidates[ci], RTLD_NOW | RTLD_LOCAL);
        if (g_drv.hip_handle) {
            if (debug) fprintf(stderr, "picograd/hip: dlopen ok %s\n", hip_candidates[ci]);
            break;
        }
        if (debug) fprintf(stderr, "picograd/hip: dlopen %s: %s\n", hip_candidates[ci], dlerror());
    }

    if (!g_drv.hip_handle) {
        if (debug) fprintf(stderr, "picograd/hip: no hip runtime found\n");
        goto fail;
    }

    for (size_t i = 0; i < sizeof(g_hip_binds)/sizeof(g_hip_binds[0]); i++) {
        void *sym = dlsym(g_drv.hip_handle, g_hip_binds[i].sym);
        if (!sym) {
            if (debug) fprintf(stderr, "picograd/hip: missing %s\n", g_hip_binds[i].sym);
            goto fail_unloaded;
        }
        *(void **)((char *)&g_drv + g_hip_binds[i].offset) = sym;
    }

    /* try to load hiprtc - optional, but needed for kernel JIT */
    const char *rtc_candidates[] = {
        "libhiprtc.so",
        "libhiprtc.so.5",
        "libhiprtc.so.6",
        "/opt/rocm/lib/libhiprtc.so",
        "/opt/rocm/lib/libhiprtc.so.5",
        "/opt/rocm/lib/hiprtc/libhiprtc.so",
        "/usr/lib/x86_64-linux-gnu/libhiprtc.so",
    };

    for (size_t ci = 0; ci < sizeof(rtc_candidates)/sizeof(rtc_candidates[0]); ci++) {
        g_drv.rtc_handle = dlopen(rtc_candidates[ci], RTLD_NOW | RTLD_LOCAL);
        if (g_drv.rtc_handle) {
            if (debug) fprintf(stderr, "picograd/hip: dlopen ok %s\n", rtc_candidates[ci]);
            break;
        }
        if (debug) fprintf(stderr, "picograd/hip: dlopen %s: %s\n", rtc_candidates[ci], dlerror());
    }

    if (g_drv.rtc_handle) {
        int rtc_ok = 1;
        for (size_t i = 0; i < sizeof(g_rtc_binds)/sizeof(g_rtc_binds[0]); i++) {
            void *sym = dlsym(g_drv.rtc_handle, g_rtc_binds[i].sym);
            if (!sym) {
                if (debug) fprintf(stderr, "picograd/hip: missing rtc %s\n", g_rtc_binds[i].sym);
                rtc_ok = 0;
                break;
            }
            *(void **)((char *)&g_drv + g_rtc_binds[i].offset) = sym;
        }
        if (!rtc_ok) {
            /* rtc is optional - clear it but keep hip */
            dlclose(g_drv.rtc_handle);
            g_drv.rtc_handle = NULL;
            memset(&g_drv.rtcCreateProgram, 0, sizeof(void*) * 7);
            if (debug) fprintf(stderr, "picograd/hip: hiprtc incomplete, running without JIT (gemm/kernels disabled)\n");
        } else if (debug) {
            fprintf(stderr, "picograd/hip: hiprtc loaded\n");
        }
    } else if (debug) {
        fprintf(stderr, "picograd/hip: hiprtc not found, running without JIT\n");
    }

    /* check device count to ensure driver is functional */
    int count = 0;
    if (g_drv.getDeviceCount(&count) != 0 || count == 0) {
        if (debug) fprintf(stderr, "picograd/hip: no devices (count=%d)\n", count);
        /* still consider driver present but init will fail later */
        /* keep g_state =0 but hip_init will check and return unsupported */
    }

    g_state = 0;
    return &g_drv;

fail_unloaded:
    if (g_drv.hip_handle) dlclose(g_drv.hip_handle);
    if (g_drv.rtc_handle) dlclose(g_drv.rtc_handle);
    memset(&g_drv, 0, sizeof(g_drv));
fail:
    g_state = 1;
    if (err) *err = PG_ERR_UNSUPPORTED;
    return NULL;
}
