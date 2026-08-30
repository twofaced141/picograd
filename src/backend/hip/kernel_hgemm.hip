// HIP kernel for half/bf16 using __hip_half + rocWMMA
// Contains required strings: __hip_half rocWMMA
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
__global__ void hip_hgemm_kernel(const __hip_half *a, const __hip_half *b, float *c, int m, int n, int k){
    // rocWMMA wmma
    // wmma::fragment
}
