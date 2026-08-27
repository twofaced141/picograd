#ifndef PICOGRAD_BACKEND_HIP_DRIVER_H
#define PICOGRAD_BACKEND_HIP_DRIVER_H

#include "../backend.h"
#include <stddef.h>

typedef struct pg_hip_drv {
    void *hip_handle;
    void *rtc_handle;
    int (*getDeviceCount)(int *count);
    int (*getDevice)(int *device);
    int (*setDevice)(int device);
    int (*malloc)(void **ptr, size_t size);
    int (*free)(void *ptr);
    int (*memcpy)(void *dst, const void *src, size_t size, int kind);
    int (*memset)(void *dst, int value, size_t size);
    int (*deviceSynchronize)(void);
    int (*moduleLoadData)(void **module, const void *image);
    int (*moduleGetFunction)(void **function, void *module, const char *kname);
    int (*moduleLaunchKernel)(void *function,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, void *stream,
                              void **kernelParams, void **extra);
    /* hiprtc */
    int (*rtcCreateProgram)(void **prog, const char *src, const char *name,
                            int numHeaders, const char **headers, const char **includeNames);
    int (*rtcCompileProgram)(void *prog, int numOptions, const char **options);
    int (*rtcGetCodeSize)(void *prog, size_t *codeSizeRet);
    int (*rtcGetCode)(void *prog, char *code);
    int (*rtcDestroyProgram)(void **prog);
    int (*rtcGetProgramLogSize)(void *prog, size_t *logSizeRet);
    int (*rtcGetProgramLog)(void *prog, char *log);
} pg_hip_drv;

const pg_hip_drv *pg_hip_drv_get(pg_status *err);

#endif
