// HipBackend.hh
#pragma once
#include <hip/hip_runtime.h>

class HipBackend {
public:
    HipBackend();
    void runTestKernel();
};
