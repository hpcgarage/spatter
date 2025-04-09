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
    size_t wrap, size_t count, size_t start_thread) 
{
    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t total_id = start_thread + thread_id;
    size_t j = total_id % pattern_length;
    size_t i = total_id / pattern_length;
    
    if (i < count) {
        size_t pattern_value = pattern[j];
        size_t sparse_index = pattern_value + delta * i;
        size_t dense_index = j + pattern_length * (i % wrap);
        
        // Bounds checking can be removed for performance if confident in size calculations
        if (sparse_index < (pattern_length * count * delta) && 
            dense_index < (pattern_length * wrap)) {
            sparse[sparse_index] = dense[dense_index];
        }
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

////////////////////////
//
//fill in here with the missing scatter wrapper:

float hip_scatter_wrapper(const size_t *pattern, double *sparse,
                          const double *dense, size_t pattern_length,
                          size_t delta, size_t wrap, size_t count) {
    // 1. Create HIP events for overall timing.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // 2. Determine threads per block.
    // Option A: Use a dynamic value based on pattern_length, capped at 1024.
    const int threads_per_block = std::min(pattern_length, static_cast<size_t>(1024));
    // Option B: Alternatively, you could set a fixed value (e.g., 256) if that suits your workload better.
    // const int threads_per_block = 256;

    // 3. Calculate the total work to be done.
    size_t total_threads = pattern_length * count;

    // 4. Query device properties.
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    // 5. Calculate a safe grid limit.
    // Although the reported grid max size is huge, practical launches on MI210 may fail when using too many blocks.
    // Here we use a heuristic divisor to scale down the theoretical limit.
    const int heuristic_divisor = 128; // Adjust this value based on your experimentation.
    int safe_max_blocks_per_grid = prop.maxGridSize[0] / heuristic_divisor;
    size_t safe_max_threads_per_grid = static_cast<size_t>(safe_max_blocks_per_grid) * threads_per_block;
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // Diagnostic print: display the computed limits.
    //printf("Workload Diagnostic:\n");
    //printf("  Total threads: %zu\n", total_threads);
    //printf("  Threads per block: %d\n", threads_per_block);
    //printf("  Theoretical max blocks per grid: %d\n", prop.maxGridSize[0]);
    //printf("  Safe max blocks per grid (heuristic): %d\n", safe_max_blocks_per_grid);
    //printf("  Safe max threads per grid: %zu\n", safe_max_threads_per_grid);
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (void*)hip_scatter);
    printf("Registers per thread: %d, Shared memory per block: %lu\n", attr.numRegs, attr.sharedSizeBytes);

    // 6. Ensure the device is ready.
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start));

    // 7. Launch the kernel in multiple iterations to cover all threads.
    size_t processed_threads = 0;
    int launch_index = 0;
    while (processed_threads < total_threads) {
        // Determine the remaining work.
        size_t remaining_threads = total_threads - processed_threads;

        // Choose threads for this launch (without exceeding safe_max_threads_per_grid).
        size_t threads_this_launch = std::min(remaining_threads, safe_max_threads_per_grid);

        // Calculate how many blocks are needed.
        int blocks_this_launch = static_cast<int>((threads_this_launch + threads_per_block - 1) / threads_per_block);
        // Ensure we do not exceed the safe maximum blocks.
        blocks_this_launch = std::min(blocks_this_launch, safe_max_blocks_per_grid);

        // The actual threads launched is blocks * threads per block (may be slightly more than needed).
        size_t actual_threads_this_launch = static_cast<size_t>(blocks_this_launch) * threads_per_block;

        // Diagnostic print for this launch.
        printf("Launch Iteration %d:\n", launch_index);
        printf("  Processed threads so far: %zu\n", processed_threads);
        printf("  Threads requested for this launch: %zu\n", threads_this_launch);
        printf("  Blocks this launch: %d\n", blocks_this_launch);
        printf("  Actual threads to launch: %zu\n", actual_threads_this_launch);

        // Calculate additional parameters for the kernel launch.
        // launch_count: how many 'pattern' repeats can be processed in this launch.
        size_t launch_count = actual_threads_this_launch / pattern_length;
        // The starting thread index in the overall workload.
        size_t launch_start_thread = processed_threads;
        // An offset into the pattern, if needed.
        size_t launch_pattern_offset = (launch_start_thread / pattern_length) % pattern_length;

        // 8. Launch the kernel.
        // Ensure your kernel has a guard for threads that exceed the actual workload.
        hipLaunchKernelGGL(hip_scatter,
            dim3(blocks_this_launch), dim3(threads_per_block), 0, 0,
            pattern + launch_pattern_offset,
            sparse + (launch_start_thread / pattern_length) * delta,
            dense + ((launch_start_thread / pattern_length) % wrap) * pattern_length,
            pattern_length, delta, wrap, launch_count, launch_start_thread % pattern_length);
        HIP_CHECK(hipGetLastError());

        // Update the total processed threads.
        processed_threads += actual_threads_this_launch;
        launch_index++;
    }

    // 9. Finalize timing.
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float time_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&time_ms, start, stop));

    // Cleanup events.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    printf("Total kernel execution time: %e seconds\n", time_ms / 1000.0);
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

  int threads_per_block = std::min(pattern_length, static_cast<size_t>(2048));
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
