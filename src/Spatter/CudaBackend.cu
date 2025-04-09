#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Configuration.hh"

__global__ void cuda_gather(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  double x;

  if (i < count) {
    // dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
    x = sparse[pattern[j] + delta * i];
    if (x == 0.5)
      dense[0] = x;
  }
}

__global__ void cuda_scatter(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
}

__global__ void cuda_scatter_atomic(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    atomicExch((unsigned long long int *)&sparse[pattern[j] + delta * i],
        __double_as_longlong(dense[j + pattern_length * (i % wrap)]));
}

__global__ void cuda_scatter_gather(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  // printf("%lu, %lu, %lu\n", total_id, j, i);
  if (i < count)
    sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
        sparse_gather[pattern_gather[j] + delta_gather * i];
}

__global__ void cuda_scatter_gather_atomic(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  // printf("%lu, %lu, %lu\n", total_id, j, i);
  if (i < count)
    atomicExch((unsigned long long int *)&sparse_scatter[pattern_scatter[j] +
                   delta_scatter * i],
        __double_as_longlong(
            sparse_gather[pattern_gather[j] + delta_gather * i]));
}

__global__ void cuda_multi_gather(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  double x;

  if (i < count) {
    // dense[j + pattern_length * (i % wrap)] =
    // sparse[pattern[pattern_gather[j]] + delta * i];
    x = sparse[pattern[pattern_gather[j]] + delta * i];
    if (x == 0.5)
      dense[0] = x;
  }
}

__global__ void cuda_multi_scatter(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    sparse[pattern[pattern_scatter[j]] + delta * i] =
        dense[j + pattern_length * (i % wrap)];
}

__global__ void cuda_multi_scatter_atomic(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    atomicExch((unsigned long long int
                       *)&sparse[pattern[pattern_scatter[j]] + delta * i],
        __double_as_longlong(dense[j + pattern_length * (i % wrap)]));
}

float cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_gather<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_scatter<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_scatter_atomic_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_scatter_atomic<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_scatter_gather<<<blocks_per_grid, threads_per_block>>>(pattern_scatter,
      sparse_scatter, pattern_gather, sparse_gather, pattern_length,
      delta_scatter, delta_gather, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_scatter_gather_atomic_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_scatter_gather_atomic<<<blocks_per_grid, threads_per_block>>>(
      pattern_scatter, sparse_scatter, pattern_gather, sparse_gather,
      pattern_length, delta_scatter, delta_gather, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_multi_gather<<<blocks_per_grid, threads_per_block>>>(pattern,
      pattern_gather, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_multi_scatter<<<blocks_per_grid, threads_per_block>>>(pattern,
      pattern_scatter, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}

float cuda_multi_scatter_atomic_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(start));

  cuda_multi_scatter_atomic<<<blocks_per_grid, threads_per_block>>>(pattern,
      pattern_scatter, sparse, dense, pattern_length, delta, wrap, count);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return time_ms;
}
