#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_gather(const size_t *pattern, const double *sparse, double *dense, const int pattern_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < pattern_length)
    dense[i] = sparse[pattern[i]];
}

__global__ void cuda_scatter(const size_t *pattern, double *sparse, const double *dense, const int pattern_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < pattern_length)
    sparse[pattern[i]] = dense[i]; 
}

void cuda_gather_wrapper(const size_t *pattern, const double *sparse, double *dense, const int pattern_length) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (pattern_length + threadsPerBlock - 1) / threadsPerBlock;
    cuda_gather<<<blocksPerGrid, threadsPerBlock>>>(pattern, sparse, dense, pattern_length);
}

void cuda_scatter_wrapper(const size_t *pattern, double *sparse, const double *dense, const int pattern_length) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (pattern_length + threadsPerBlock - 1) / threadsPerBlock;
    cuda_scatter<<<blocksPerGrid, threadsPerBlock>>>(pattern, sparse, dense, pattern_length);
}

