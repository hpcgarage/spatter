#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_gather(const size_t *pattern, const double *sparse,
    double *dense, const int gsops, const int dense_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < gsops)
    dense[i % dense_length] = sparse[pattern[i]];
}

__global__ void cuda_scatter(const size_t *pattern, double *sparse,
    const double *dense, const int gsops, const int dense_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < gsops)
    sparse[pattern[i]] = dense[i % dense_length];
}

__global__ void cuda_scatter_gather(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const int gsops) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < gsops)
    sparse_scatter[pattern_scatter[i]] = sparse_gather[pattern_gather[i]];
}

__global__ void cuda_multi_gather(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const int gsops, const int dense_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < gsops)
    dense[i % dense_length] = sparse[pattern_gather[i]];
}

__global__ void cuda_multi_scatter(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const int gsops, const int dense_length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < gsops)
    sparse[pattern_scatter[i]] = dense[i % dense_length];
}

void cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const int pattern_length, const size_t count,
    const size_t wrap) {
  int gsops = (int)((int)pattern_length * (int)count);
  int dense_length = pattern_length * (int)wrap;

  int threads_per_block = 256;
  int blocks_per_grid = (gsops + threads_per_block - 1) / threads_per_block;
  cuda_gather<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, gsops, dense_length);
}

void cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const int pattern_length, const size_t count,
    const size_t wrap) {
  int gsops = (int)((int)pattern_length * (int)count);
  int dense_length = pattern_length * (int)wrap;

  int threads_per_block = 256;
  int blocks_per_grid = (gsops + threads_per_block - 1) / threads_per_block;
  cuda_scatter<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, gsops, dense_length);
}

void cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const int pattern_length, const size_t count,
    const size_t wrap) {
  int gsops = (int)((int)pattern_length * (int)count);

  int threads_per_block = 256;
  int blocks_per_grid = (gsops + threads_per_block - 1) / threads_per_block;
  cuda_scatter_gather<<<blocks_per_grid, threads_per_block>>>(
      pattern_scatter, sparse_scatter, pattern_gather, sparse_gather, gsops);
}

void cuda_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const int pattern_length, const size_t count, const size_t wrap) {
  int gsops = (int)((int)pattern_length * (int)count);
  int dense_length = pattern_length * (int)wrap;

  int threads_per_block = 256;
  int blocks_per_grid = (gsops + threads_per_block - 1) / threads_per_block;
  cuda_multi_gather<<<blocks_per_grid, threads_per_block>>>(
      pattern, pattern_gather, sparse, dense, gsops, dense_length);
}

void cuda_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const int pattern_length, const size_t count, const size_t wrap) {
  int gsops = (int)((int)pattern_length * (int)count);
  int dense_length = pattern_length * (int)wrap;

  int threads_per_block = 256;
  int blocks_per_grid = (gsops + threads_per_block - 1) / threads_per_block;
  cuda_multi_scatter<<<blocks_per_grid, threads_per_block>>>(
      pattern, pattern_scatter, sparse, dense, gsops, dense_length);
}
