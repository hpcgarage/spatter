#include <stdio.h>
#include <hip/hip_runtime.h>  // Replaced CUDA headers
#include "Configuration.hip.hh"

//----------------------------------------------------------
// Kernels (Direct HIP Port)
//----------------------------------------------------------

__global__ void hip_gather(const size_t *pattern, const double *sparse,
    double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;  // Removed redundant casts
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    double x = sparse[pattern[j] + delta * i];
    if (x == 0.5) dense[0] = x;  // Logic unchanged
  }
}

__global__ void hip_scatter(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
  }
}

__global__ void hip_scatter_atomic(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    atomicExch(  // HIP-compatible atomic-- manual change from Exchange to Exch
      reinterpret_cast<unsigned long long*>(&sparse[pattern[j] + delta * i]),
      __double_as_longlong(dense[j + pattern_length * (i % wrap)])
    );
  }
}

__global__ void hip_scatter_gather(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
      sparse_gather[pattern_gather[j] + delta_gather * i];
  }
}

__global__ void hip_scatter_gather_atomic(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    atomicExch(  // HIP atomic manual change to Exch from Exchange
      reinterpret_cast<unsigned long long*>(&sparse_scatter[pattern_scatter[j] + delta_scatter * i]),
      __double_as_longlong(sparse_gather[pattern_gather[j] + delta_gather * i])
    );
  }
}

__global__ void hip_multi_gather(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    double x = sparse[pattern[pattern_gather[j]] + delta * i];
    if (x == 0.5) dense[0] = x;
  }
}
// ===================== KERNELS ===================== 
__global__ void hip_multi_scatter(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;  // Simplified
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    sparse[pattern[pattern_scatter[j]] + delta * i] = 
      dense[j + pattern_length * (i % wrap)];
  }
}

__global__ void hip_multi_scatter_atomic(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    atomicExch(  // HIP atomic manually changed again
      reinterpret_cast<unsigned long long*>(&sparse[pattern[pattern_scatter[j]] + delta * i]),
      __double_as_longlong(dense[j + pattern_length * (i % wrap)])
    );
  }
}

// ===================== WRAPPERS ===================== 
float hip_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_gather, 
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, sparse, dense, pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}

float hip_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_scatter, 
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, sparse, dense, pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}
// ===================== ATOMIC SCATTER WRAPPER ===================== 
float hip_scatter_atomic_wrapper(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_scatter_atomic, 
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, sparse, dense, pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}

// ===================== SCATTER-GATHER WRAPPER ===================== 
float hip_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_scatter_gather,
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern_scatter, sparse_scatter, pattern_gather, sparse_gather,
      pattern_length, delta_scatter, delta_gather, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}

// ===================== ATOMIC SCATTER-GATHER WRAPPER ===================== 
float hip_scatter_gather_atomic_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_scatter_gather_atomic,
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern_scatter, sparse_scatter, pattern_gather, sparse_gather,
      pattern_length, delta_scatter, delta_gather, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}
// ===================== MULTI-GATHER WRAPPER ===================== 
float hip_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_multi_gather,
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, pattern_gather, sparse, dense,
      pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}

// ===================== MULTI-SCATTER WRAPPER ===================== 
float hip_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_multi_scatter,
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, pattern_scatter, sparse, dense,
      pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}

// ===================== ATOMIC MULTI-SCATTER WRAPPER ===================== 
float hip_multi_scatter_atomic_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap, size_t count) 
{
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
  int blocks_per_grid = (pattern_length * count + threads_per_block - 1) / threads_per_block;

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));

  hipLaunchKernelGGL(hip_multi_scatter_atomic,
      dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
      pattern, pattern_scatter, sparse, dense,
      pattern_length, delta, wrap, count
  );
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float time_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return time_ms;
}
