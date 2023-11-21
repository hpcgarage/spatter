#ifndef _GNU_SOURCE
    #define _GNU_SOURCE //needed for string.h to include strcasestr
#endif
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sgbuf.h"
#include "cuda-jit-backend.h"
#include "../include/parse-args.h"

#include <curand_kernel.h>

/////////////
// GLOBALS //
/////////////

#define typedef uint unsigned long
__device__ int final_block_idx_dev = -1;
__device__ int final_thread_idx_dev = -1;
__device__ double final_gather_data_dev = -1;

/////////////
// UTILITY //
/////////////

// See: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

void create_dev_buffers_cuda(sgDataBuf* source)
{
    cudaError_t ret;
    ret = cudaMalloc((void **)&(source->dev_ptr_cuda), source->size);
    if (ret != cudaSuccess) {
        printf("Could not allocate gpu memory (%zu bytes): %s\n", source->size, cudaGetErrorName(ret));
        exit(1);
    }
}

int find_device_cuda(char *name) {
    if (!name) {
        return -1;
    }
    int nDevices = 0;
    struct cudaDeviceProp prop;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaGetDeviceProperties(&prop, i);
        if (strcasestr(prop.name, name)){
            return i;
        }
    }
    return -1;
}

/////////////
// KERNELS //
/////////////


//////////////
// WRAPPERS //
//////////////
double cuda_jit_wrapper(struct run_config rc)
{
    printf("Cuda JIT!\n");
}

//template<size_t idx_len, size_t delta, size_t tt>
__global__ void gather_big(double *src, ssize_t* idx)
{

}

/*
// Multiple blocks per GSOP
__global__ void gather_big(double *src, ssize_t* idx, size_t idx_len, size_t delta)
{
    extern __shared__ ssize_t idx_shared[];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    int blocks_per_gsop = blockDim.x/idx_len + (blockDim.x%idx_len==0?0:1); // Guaranteed to be > 1
    int gatherid = blockDim.x/blocks_per_gsop; // which gather we are working on
    int gatheroffset = bid % blocks_per_gsop; // which portion of the gather we are working on

    int last = (gatheroffset == blocks_per_gsop-1) && (idx_len%blockDim.x!=0); // only need to treat last gather special if it isn't the same size as others
    //printf("%d %d\n", gatheroffset == blocks_per_gsop-1, blockDim.x%idx_len==0);

    //printf("%d %d %d %lu\n", gatheroffset, blocks_per_gsop, blockDim.x, idx_len);
    if (last) {
        //printf("LAST 1\n");
        // Last block needs to copy idx_len - (blockDim.x*(blocks_per_gsop-1))
        if ( tid < (idx_len - (blockDim.x*(blocks_per_gsop-1))) ) {
            idx_shared[tid] = idx[(blockDim.x)*(blocks_per_gsop-1)+tid];
        }
    } else {
        idx_shared[tid] = idx[gatheroffset*blockDim.x + tid];
    }

    double *src_loc = src + delta*gatherid + idx_len*gatheroffset;
    double x;
    // Only do the read if...
    if (last) {
        //printf("LAST 2\n");
        if ( tid < (idx_len - (blockDim.x*(blocks_per_gsop-1))) ) {
            idx_shared[tid] = idx[(blockDim.x)*(blocks_per_gsop-1)+tid];
            x = src_loc[idx_shared[tid]];
        }
    } else {
        x = src_loc[idx_shared[tid]];
    }
    if (x==0.5) src[0] = x; // dummy computation

}
*/

// 1 or multiple GSOPs per block
/*
__global__ void gather_small(double *src, ssize_t* idx, size_t idx_len, size_t delta, char validate)
{
    __shared__ int idx_shared[];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    int gsop_per_block = idx_len/blockDim.x;

    // Load the idx buffer from gloabl to shared memory
    // The size of the block may be larger than the pattern,
    // so we need to make sure we don't copy too much.
    if (tid < idx_len) {
        idx_shared[tid] = idx[tid];
    }

    // If the pattern is much smaller than the thread block size,
    // we will pack more than one gather into a thread block
    int ngatherperblock = blockDim.x / idx_len;
    int gatherid = tid / idx_len;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;
    double x;
    x = src_loc[idx_shared[tid%V]];
    if (x==0.5) src[0] = x; // dummy computation

}
*/


extern "C" float cuda_block_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        ssize_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap,
        int wpt,
        size_t morton,
        uint32_t *order,
        uint32_t *order_dev,
        int stride,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate)
{

    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    //if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    gpuErrchk( cudaMemcpy(pat_dev, pat, sizeof(sgIdx_t)*pat_len, cudaMemcpyHostToDevice) );
    if (morton) {
        gpuErrchk( cudaMemcpy(order_dev, order, sizeof(uint32_t)*n, cudaMemcpyHostToDevice) );
    }

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );
    // KERNEL
    if (kernel == GATHER) {
        if (pat_len/block[0] <= 1) {
            //gather_block<<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate, pat_len);
        } else {
            //printf("CALLING GATHER BIG\n");
            gather_big<<<grid_dim, block_dim,block[0]*sizeof(ssize_t)>>>(source, pat_dev);
        }
        gpuErrchk( cudaMemcpyFromSymbol(final_gather_data, final_gather_data_dev, sizeof(double), 0, cudaMemcpyDeviceToHost) );
    }

    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    gpuErrchk( cudaMemcpyFromSymbol(final_block_idx, final_block_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpyFromSymbol(final_thread_idx, final_thread_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );

    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms;

}


