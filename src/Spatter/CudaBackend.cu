#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_gather(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  double x;

  if (j < pattern_length && i < count) {
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

  if (j < pattern_length && i < count)
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
  if (j < pattern_length && i < count)

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

  if (j < pattern_length && i < count) {
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

  if (j < pattern_length && i < count)
    atomicExch((unsigned long long int
                       *)&sparse[pattern[pattern_scatter[j]] + delta * i],
        __double_as_longlong(dense[j + pattern_length * (i % wrap)]));
}

float cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  cuda_gather<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, pattern_length, delta, wrap, count);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

float cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  cuda_scatter<<<blocks_per_grid, threads_per_block>>>(
      pattern, sparse, dense, pattern_length, delta, wrap, count);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

float cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  cuda_scatter_gather<<<blocks_per_grid, threads_per_block>>>(pattern_scatter,
      sparse_scatter, pattern_gather, sparse_gather, pattern_length,
      delta_scatter, delta_gather, wrap, count);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

float cuda_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  cuda_multi_gather<<<blocks_per_grid, threads_per_block>>>(pattern,
      pattern_gather, sparse, dense, pattern_length, delta, wrap, count);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

float cuda_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int threads_per_block = min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  cuda_multi_scatter<<<blocks_per_grid, threads_per_block>>>(pattern,
      pattern_scatter, sparse, dense, pattern_length, delta, wrap, count);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}
