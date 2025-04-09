#include <stdio.h>
#include <hip/hip_runtime.h>  
#include "Configuration.hip.hh"


__global__ void hip_gather(const size_t *pattern, const double *sparse,
    double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count) 
{
  size_t total_id = blockDim.x * blockIdx.x + threadIdx.x; // check cast change 
  size_t j = total_id % pattern_length;
  size_t i = total_id / pattern_length;

  if (i < count) {
    double x = sparse[pattern[j] + delta * i];
    if (x == 0.5) dense[0] = x;  
  }
}

__global__ void hip_scatter(const size_t *pattern, double *sparse,
                          const double *dense, const size_t pattern_length,
                          const size_t delta, const size_t wrap, const size_t count) {
  size_t total_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = total_id % pattern_length; // pattern index
  size_t i = total_id / pattern_length; // count index
  if (i < count) {
    sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
  }
}
// handles multiple launches for large scatters
__global__ void hip_scatter_offset(const size_t *pattern, double *sparse,
                                   const double *dense, size_t pattern_length,
                                   size_t delta, size_t wrap, size_t count,
                                   size_t offset) {
    size_t total_id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t j = total_id % pattern_length;
    size_t i = total_id / pattern_length;

    if (i < count) {
        size_t global_i = i + offset;
        size_t sparse_idx = pattern[j] + delta * global_i;
        size_t dense_idx = j + pattern_length * (global_i % wrap);
        sparse[sparse_idx] = dense[dense_idx];
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
    atomicExch(  
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
    atomicExch(  
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
    atomicExch(  
      reinterpret_cast<unsigned long long*>(&sparse[pattern[pattern_scatter[j]] + delta * i]),
      __double_as_longlong(dense[j + pattern_length * (i % wrap)])
    );
  }
}
 
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

float hip_scatter_wrapper(const size_t *d_pattern, double *d_sparse,
                          const double *d_dense, size_t pattern_length,
                          size_t delta, size_t wrap, size_t count) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    int maxGridSizeX;
    hipDeviceGetAttribute(&maxGridSizeX, hipDeviceAttributeMaxGridDimX, 0);

    int wavefrontSize;
    hipDeviceGetAttribute(&wavefrontSize, hipDeviceAttributeWarpSize, 0);

    size_t wavefrontFactor = wavefrontSize * 2; 
    size_t maxBlocksPerLaunch = maxGridSizeX / wavefrontFactor;

    int threadsPerBlock = 0;
    int minGridSize = 0;
    hipOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock,
                                      hip_scatter, 0, 0);
    threadsPerBlock = std::min(threadsPerBlock, 1024);
    threadsPerBlock = std::min((int)pattern_length, threadsPerBlock);

    size_t total_threads = pattern_length * count;
    size_t blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    hipDeviceSynchronize();
    hipEventRecord(start);

    if (blocksPerGrid <= maxBlocksPerLaunch) {
        hipLaunchKernelGGL(hip_scatter, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                           d_pattern, d_sparse, d_dense, pattern_length,
                           delta, wrap, count);
        hipDeviceSynchronize();
    } else {
        size_t maxThreadsPerLaunch = maxBlocksPerLaunch * threadsPerBlock;
        size_t countsPerLaunch = maxThreadsPerLaunch / pattern_length;
        size_t launches = (count + countsPerLaunch - 1) / countsPerLaunch;

        for (size_t i = 0; i < launches; ++i) {
            size_t offset = i * countsPerLaunch;
            size_t counts_this = std::min(countsPerLaunch, count - offset);
            size_t threads_this = pattern_length * counts_this;
            size_t blocks_this = (threads_this + threadsPerBlock - 1) / threadsPerBlock;

            hipLaunchKernelGGL(hip_scatter_offset, dim3(blocks_this), dim3(threadsPerBlock), 0, 0,
                              d_pattern, d_sparse, d_dense, pattern_length,
                              delta, wrap, counts_this, offset);
            hipDeviceSynchronize();
        }
    }

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float time_ms = 0.0f;
    hipEventElapsedTime(&time_ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return time_ms;
}
 
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
