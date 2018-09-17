#ifndef SG_CUDA_H
#define SG_CUDA_H
#include <cuda.h>
__global__ void my_kernel();
__global__ void scatter(double* target, double* source, long* ti, long* si, long ot, long os, long oi);
#endif
