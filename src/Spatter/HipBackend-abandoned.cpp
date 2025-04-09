#include <stdio.h>
#include <hip/hip_runtime.h>  // Replaced <cuda_runtime.h>
#include "Configuration.hh"

__global__ void hip_gather(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx
  double x;
  if (i < count) {
    x = sparse[pattern[j] + delta * i];
    if (x == 0.5)
      dense[0] = x;
  }
}

__global__ void hip_scatter(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx
  if (i < count)
    sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
}

__global__ void hip_scatter_atomic(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx
  if (i < count) {
    // HIP uses atomicExchange with the same casting as CUDA
    atomicExchange(
        (unsigned long long int *)&sparse[pattern[j] + delta * i],
        __double_as_longlong(dense[j + pattern_length * (i % wrap)])
    );
  }
}
__global__ void hip_scatter_gather(
    const size_t *pattern_scatter, double *sparse_scatter,
    const size_t *pattern_gather, const double *sparse_gather,
    const size_t pattern_length, const size_t delta_scatter,
    const size_t delta_gather, const size_t wrap, const size_t count) 
{
    // ... identical kernel body to CUDA version ...
}

__global__ void hip_scatter_gather_atomic(
    const size_t *pattern_scatter, double *sparse_scatter,
    const size_t *pattern_gather, const double *sparse_gather,
    const size_t pattern_length, const size_t delta_scatter,
    const size_t delta_gather, const size_t wrap, const size_t count) 
{
    size_t total_id = (size_t)(blockDim.x * blockIdx.x + threadIdx.x);
    size_t j = total_id % pattern_length;
    size_t i = total_id / pattern_length;
    
    if (i < count) {
        // HIP atomicExch has identical signature to CUDA
        atomicExchange(  // Changed from atomicExch
            (unsigned long long int*)&sparse_scatter[pattern_scatter[j] + delta_scatter * i],
            __double_as_longlong(sparse_gather[pattern_gather[j] + delta_gather * i])
        );
    }
}
