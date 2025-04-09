// HipBackend.cpp
#include "HipBackend.hh"
#include <iostream>

__global__ void testKernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

HipBackend::HipBackend() {
    // Constructor - could initialize HIP here if needed
}

void HipBackend::runTestKernel() {
    void* args[] = {};
    hipError_t err = hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(testKernel),
        1, 1, 1,  // grid dims
        16, 1, 1, // block dims
        0,        // shared mem
        0,        // stream
        args,
        nullptr   // extra
    );
    
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
    }
    hipDeviceSynchronize();
}
