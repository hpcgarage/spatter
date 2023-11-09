#ifndef _GNU_SOURCE
    #define _GNU_SOURCE //needed for string.h to include strcasestr
#endif
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sgbuf.h"
#include "cuda-backend.h"
#include "../include/parse-args.h"

#include <curand_kernel.h>

#define typedef uint unsigned long

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


//__device__ int dummy = 0;
__device__ int final_block_idx_dev = -1;
__device__ int final_thread_idx_dev = -1;
__device__ double final_gather_data_dev = -1;

template<int v>
__global__ void scatter_t(double* target,
                        double* source,
                        long* ti,
                        long* si)
{
    extern __shared__ char space[];

    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);

    double buf[v];
    long idx[v];

    for(int i = 0; i < v; i++){
        buf[i] = source[gid+i];
    }

    for(int i = 0; i < v; i++){
       idx[i] = ti[gid+i];
    }

    for(int i = 0; i < v; i++){
        target[idx[i]] = buf[i];
    }
}

template<int v>
__global__ void gather_t(double* target,
                        double* source,
                        long* ti,
                        long* si)
{
    extern __shared__ char space[];

    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);
    double buf[v];

    for(int i = 0; i < v; i++){
        buf[i] = source[si[gid+i]];
    }

    for(int i = 0; i < v; i++){
        target[gid+i] = buf[i];
    }

}

template<int v>
__global__ void sg_t(double* target,
                    double* source,
                    long* ti,
                    long* si)
{
    extern __shared__ char space[];

    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);
    long sidx[v];
    long tidx[v];

    for(int i = 0; i < v; i++){
        sidx[i] = si[gid+i];
    }
    for(int i = 0; i < v; i++){
        tidx[i] = ti[gid+i];
    }
    for(int i = 0; i < v; i++){
        target[tidx[i]] = source[sidx[i]];
    }

}

extern "C" int translate_args(unsigned int dim, unsigned int* grid, unsigned int* block, dim3 *grid_dim, dim3 *block_dim){
    if (!grid || !block || dim == 0 || dim > 3) {
        return 1;
    }
    if (dim == 1) {
        *grid_dim  = dim3(grid[0]);
        *block_dim = dim3(block[0]);
    }else if (dim == 2) {
        *grid_dim  = dim3(grid[0],  grid[1]);
        *block_dim = dim3(block[0], block[1]);
    }else if (dim == 3) {
        *grid_dim  = dim3(grid[0],  grid[1],  grid[2]);
        *block_dim = dim3(block[0], block[1], block[2]);
    }
    return 0;

}

extern "C" float cuda_sg_wrapper(enum sg_kernel kernel,
                                size_t vector_len,
                                uint dim, uint* grid, uint* block,
                                double* target, double *source,
                                long* ti, long* si,
                                unsigned int shmem){
    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );

    if(kernel == SCATTER)
    {
        if (vector_len == 1)
            scatter_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            scatter_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            scatter_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            scatter_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            scatter_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            scatter_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            scatter_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            scatter_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 128)
            scatter_t<128><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 256)
            scatter_t<256><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 512)
            scatter_t<512><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 1024)
            scatter_t<1024><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2048)
            scatter_t<2048><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4096)
            scatter_t<4096><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GATHER)
    {
        if (vector_len == 1)
            gather_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            gather_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            gather_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            gather_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            gather_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            gather_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            gather_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            gather_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 128)
            gather_t<128><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 256)
            gather_t<256><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 512)
            gather_t<512><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 1024)
            gather_t<1024><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2048)
            gather_t<2048><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4096)
            gather_t<4096><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GS)
    {
        if (vector_len == 1)
            sg_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            sg_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            sg_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            sg_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            sg_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            sg_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            sg_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            sg_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 128)
            sg_t<128><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 256)
            sg_t<256><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 512)
            sg_t<512><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 1024)
            sg_t<1024><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2048)
            sg_t<2048><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4096)
            sg_t<4096><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else
    {
        printf("ERROR UNRECOGNIZED KERNEL\n");
        exit(1);
    }
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms;

}

//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__global__ void scatter_block(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate)
{
    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;

    //for (int i = 0; i < wpb; i++) {
        src_loc[idx_shared[tid%V]] = idx_shared[tid%V];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}
}

//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__global__ void scatter_block_random(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n)
{
    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;
    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    unsigned long long sequence = blockIdx.x; //all thread blocks can use same sequence
    unsigned long long offset = gatherid;
    curandState_t state;
    curand_init(seed, sequence, offset, &state);//everyone with same gather id should get same src_loc

    int random_gatherid = (int)(n * curand_uniform(&state));


    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }


    double *src_loc = src + (bid*ngatherperblock+random_gatherid)*delta;

    //for (int i = 0; i < wpb; i++) {
        src_loc[idx_shared[tid%V]] = idx_shared[tid%V];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}
}

//V2 = 8
//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__global__ void gather_block(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate)
{
    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;
    
    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = src_loc[idx_shared[tid%V]];
        return;
    }
    #endif
    
    double x;

    //for (int i = 0; i < wpb; i++) {
        x = src_loc[idx_shared[tid%V]];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

template<int V>
__global__ void gather_block_morton(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, uint32_t *order, char validate)
{
    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+order[gatherid])*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = src_loc[idx_shared[tid%V]];
        return;
    }
    #endif

    double x;

    //for (int i = 0; i < wpb; i++) {
        x = src_loc[idx_shared[tid%V]];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

template<int V>
__global__ void gather_block_stride(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, int stride, char validate)
{
    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = src_loc[stride*(tid%V)];
    }
    #endif

    double x;

    //for (int i = 0; i < wpb; i++) {
    x = src_loc[stride*(tid%V)];
    //src_loc[idx_shared[tid%V]] = 1337.;
    //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

template<int V>
__global__ void gather_block_random(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n)
{

    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;
    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    unsigned long long sequence = blockIdx.x; //all thread blocks can use same sequence
    unsigned long long offset = gatherid;
    curandState_t state;
    curand_init(seed, sequence, offset, &state);//everyone with same gather id should get same src_loc
    int random_gatherid = (int)(n * curand_uniform(&state));


    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    double *src_loc = src + (bid*ngatherperblock+random_gatherid)*delta;
    double x;

    //for (int i = 0; i < wpb; i++) {
        x = src_loc[idx_shared[tid%V]];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

//todo -- add WRAP
template<int V>
__global__ void gather_new(double* source,
                        sgIdx_t* idx, size_t delta, int dummy, int wpt)
{
    __shared__ int idx_shared[V];

    int tid  = threadIdx.x;
    //int bid  = blockIdx.x;
    //int nblk = blockDim.x;

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    double *sl = source + wpt*gid*delta;

    double buf[V];

    for (int j = 0; j < wpt; j++) {
        for (int i = 0; i < V; i++) {
            buf[i] = sl[idx_shared[i]];
            //source[i+gid*delta] = 8;
            //sl[i] = sl[idx[i]];
        }
        sl = sl + delta;
    }

    if (dummy) {
        sl[idx_shared[0]] = buf[dummy];
    }

    /*
    for (int i = 0; i < V; i++) {
        if (buf[i] == 199402) {
            printf("oop\n");
        }
    }
    */

        //printf("idx[1]: %d\n", idx[1]);
        /*
        for (int i = 0; i < V; i++) {
            printf("idx %d is %zu", i, idx[i]);
        }
        printf("\n");
        */

}

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

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    gpuErrchk( cudaMemcpy(pat_dev, pat, sizeof(sgIdx_t)*pat_len, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(order_dev, order, sizeof(uint32_t)*n, cudaMemcpyHostToDevice) );

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );
    // KERNEL
    if (kernel == GATHER) {
        if (morton) {
            if (pat_len == 8) {
                gather_block_morton<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 16) {
                gather_block_morton<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 32) {
                gather_block_morton<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 64) {
                gather_block_morton<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 73) {
                gather_block_morton<73><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 128) {
                gather_block_morton<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 256) {
                gather_block_morton<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 512) {
                gather_block_morton<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 1024) {
                gather_block_morton<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 2048) {
                gather_block_morton<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else if (pat_len == 4096) {
                gather_block_morton<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, order_dev, validate);
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }

        } else if (stride >= 0) {
            if (pat_len == 8) {
                gather_block_stride<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 16) {
                gather_block_stride<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 32) {
                gather_block_stride<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 64) {
                gather_block_stride<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 73) {
                gather_block_stride<73><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 128) {
                gather_block_stride<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 256) {
                gather_block_stride<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 512) {
                gather_block_stride<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 1024) {
                gather_block_stride<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 2048) {
                gather_block_stride<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else if (pat_len == 4096) {
                gather_block_stride<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, stride, validate);
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }

        } else {
            if (pat_len == 8) {
                gather_block<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 16) {
                gather_block<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 32) {
                gather_block<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 64) {
                gather_block<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 73) {
                gather_block<73><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 128) {
                gather_block<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 256) {
                gather_block<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 512) {
                gather_block<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 1024) {
                gather_block<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 2048) {
                gather_block<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else if (pat_len == 4096) {
                gather_block<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }
        }
        gpuErrchk( cudaMemcpyFromSymbol(final_gather_data, final_gather_data_dev, sizeof(double), 0, cudaMemcpyDeviceToHost) );
    } else if (kernel == SCATTER) {
        if (pat_len == 8) {
            scatter_block<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 16) {
            scatter_block<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 32) {
            scatter_block<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 64) {
            scatter_block<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 128) {
            scatter_block<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len ==256) {
            scatter_block<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 512) {
            scatter_block<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 1024) {
            scatter_block<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len == 2048) {
            scatter_block<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else if (pat_len ==4096) {
            scatter_block<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, validate);
        }else {
            printf("ERROR NOT SUPPORTED, %zu\n", pat_len);
            exit(1);
        }

    }

    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    gpuErrchk( cudaMemcpyFromSymbol(final_block_idx, final_block_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpyFromSymbol(final_thread_idx, final_thread_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );


    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms;


}

extern "C" float cuda_block_random_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        ssize_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap,
        int wpt, size_t seed)
{
    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    gpuErrchk( cudaMemcpy(pat_dev, pat, sizeof(sgIdx_t)*pat_len, cudaMemcpyHostToDevice) );

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );
    // KERNEL
    if (kernel == GATHER) {
        if (pat_len == 8) {
            gather_block_random<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 16) {
            gather_block_random<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 32) {
            gather_block_random<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 64) {
            gather_block_random<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 128) {
            gather_block_random<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len ==256) {
            gather_block_random<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 512) {
            gather_block_random<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 1024) {
            gather_block_random<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 2048) {
            gather_block_random<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len ==4096) {
            gather_block_random<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else {
            printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
        }
    } else if (kernel == SCATTER) {
        if (pat_len == 8) {
            scatter_block_random<8><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 16) {
            scatter_block_random<16><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 32) {
            scatter_block_random<32><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 64) {
            scatter_block_random<64><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 128) {
            scatter_block_random<128><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len ==256) {
            scatter_block_random<256><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 512) {
            scatter_block_random<512><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 1024) {
            scatter_block_random<1024><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len == 2048) {
            scatter_block_random<2048><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else if (pat_len ==4096) {
            scatter_block_random<4096><<<grid_dim, block_dim>>>(source, pat_dev, pat_len, delta, wpt, seed, n);
        }else {
            printf("ERROR NOT SUPPORTED, %zu\n", pat_len);
        }

    }
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );


    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms;


}


template<int V>
__global__ void sg_block(double *source, double* target, sgIdx_t* pat_gath, sgIdx_t* pat_scat, spSize_t pat_len, size_t delta_gather, size_t delta_scatter, int wpt, char validate)
{
    __shared__ int idx_gath[V];
    __shared__ int idx_scat[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx_gath[tid] = pat_gath[tid];
        idx_scat[tid] = pat_scat[tid];
    }

    int ngatherperblock = blockDim.x / V;
    int nscatterperblock = ngatherperblock;

    int gatherid = tid / V;
    int scatterid = gatherid;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta_gather;
    double *target_loc = target + (bid*nscatterperblock+scatterid)*delta_scatter;
    
    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[idx_gath[tid%V]];
        return;
    }
    #endif
    
    target_loc[idx_scat[tid%V]] = source_loc[idx_gath[tid%V]];
}

extern "C" float cuda_block_sg_wrapper(uint dim, uint* grid, uint* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* pat_gath_dev,
        sgIdx_t* pat_scat_dev,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate)
{
    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    gpuErrchk( cudaMemcpy(pat_gath_dev, rc->pattern_gather, sizeof(sgIdx_t)*rc->pattern_gather_len, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(pat_scat_dev, rc->pattern_scatter, sizeof(sgIdx_t)*rc->pattern_scatter_len, cudaMemcpyHostToDevice) );

    size_t delta_gather = rc->delta_gather;
    size_t delta_scatter = rc->delta_scatter;

    spSize_t pat_len = rc->pattern_gather_len;

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );
   
    // KERNEL
    if (pat_len == 8) {
        sg_block<8><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 16) {
        sg_block<16><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 32) {
        sg_block<32><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 64) {
        sg_block<64><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 73) {
        sg_block<73><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 128) {
        sg_block<128><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 256) {
        sg_block<256><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 512) {
        sg_block<512><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 1024) {
        sg_block<1024><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 2048) {
        sg_block<2048><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else if (pat_len == 4096) {
        sg_block<4096><<<grid_dim, block_dim>>>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate);
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel launch: %d (%s)\n", err, cudaGetErrorString(err));
        printf("Parameters were: pat_len: %zu, grid_dim: (%d,%d,%d), block_dim: (%d,%d,%d)\n", pat_len, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
        exit(1);

    }

    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    gpuErrchk( cudaMemcpyFromSymbol(final_block_idx, final_block_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpyFromSymbol(final_thread_idx, final_thread_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );


    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms; 
}


template<int V>
__global__ void multiscatter_block(double *source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate)
{
    __shared__ int idx[V];
    __shared__ int idx_scat[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx[tid] = outer_pat[tid];
        idx_scat[tid] = inner_pat[tid];
    }

    int ngatherperblock = blockDim.x / V;

    int gatherid = tid / V;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[tid%V];
        return;
    }
    #endif

    source_loc[idx[idx_scat[tid%V]]] = target[tid%V];
}

extern "C" float cuda_block_multiscatter_wrapper(uint grid, uint block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate)
{
    cudaEvent_t start, stop;
    gpuErrchk( cudaMemcpy(outer_pat, rc->pattern, sizeof(sgIdx_t)*rc->pattern_len, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(inner_pat, rc->pattern_scatter, sizeof(sgIdx_t)*rc->pattern_scatter_len, cudaMemcpyHostToDevice) );

    size_t delta = rc->delta;
    spSize_t pat_len = rc->pattern_scatter_len;

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );

    // KERNEL
    if (pat_len == 8) {
        multiscatter_block<8><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 16) {
        multiscatter_block<16><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 32) {
        multiscatter_block<32><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 64) {
        multiscatter_block<64><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 73) {
        multiscatter_block<73><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 128) {
        multiscatter_block<128><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 256) {
        multiscatter_block<256><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 512) {
        multiscatter_block<512><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 1024) {
        multiscatter_block<1024><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 2048) {
        multiscatter_block<2048><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 4096) {
        multiscatter_block<4096><<<grid, block>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    // Check for errors in kernel lauch
    gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() ); //TODO: When should we synchronize?
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    gpuErrchk( cudaMemcpyFromSymbol(final_block_idx, final_block_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpyFromSymbol(final_thread_idx, final_thread_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );

    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms;
}

template<int V>
__global__ void multigather_block(double *source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate)
{
    __shared__ int idx[V];
    __shared__ int idx_gath[V];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx[tid] = outer_pat[tid];
        idx_gath[tid] = inner_pat[tid];
    }

    int ngatherperblock = blockDim.x / V;

    int gatherid = tid / V;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta;
    
    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[idx[idx_gath[tid%V]]];
        return;
    }
    #endif
    
    target[tid%V] = source_loc[idx[idx_gath[tid%V]]];
}

extern "C" float cuda_block_multigather_wrapper(uint dim, uint* grid, uint* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate)
{
    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    gpuErrchk( cudaMemcpy(outer_pat, rc->pattern, sizeof(sgIdx_t)*rc->pattern_len, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(inner_pat, rc->pattern_gather, sizeof(sgIdx_t)*rc->pattern_gather_len, cudaMemcpyHostToDevice) );

    size_t delta = rc->delta;

    spSize_t pat_len = rc->pattern_gather_len;

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaEventRecord(start) );
   
    // KERNEL
    if (pat_len == 8) {
        multigather_block<8><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 16) {
        multigather_block<16><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 32) {
        multigather_block<32><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 64) {
        multigather_block<64><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 73) {
        multigather_block<73><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 128) {
        multigather_block<128><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 256) {
        multigather_block<256><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 512) {
        multigather_block<512><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 1024) {
        multigather_block<1024><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 2048) {
        multigather_block<2048><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else if (pat_len == 4096) {
        multigather_block<4096><<<grid_dim, block_dim>>>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate);
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));        
     
    gpuErrchk( cudaMemcpyFromSymbol(final_block_idx, final_block_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpyFromSymbol(final_thread_idx, final_thread_idx_dev, sizeof(int), 0, cudaMemcpyDeviceToHost) );


    float time_ms = 0;
    gpuErrchk( cudaEventElapsedTime(&time_ms, start, stop) );
    return time_ms; 
}


#define INSTANTIATE(V)\
template __global__ void scatter_t<V>(double* target, double* source, long* ti, long* si);\
template __global__ void gather_t<V>(double* target, double* source, long* ti, long* si); \
template __global__ void sg_t<V>(double* target, double* source, long* ti, long* si);\
template __global__ void gather_new<V>(double* source, sgIdx_t* idx, size_t delta, int dummy, int wpt); \
template __global__ void gather_block<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate);\
template __global__ void gather_block_morton<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, uint32_t *order, char validate);\
template __global__ void gather_block_stride<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, int stride, char validate);\
template __global__ void scatter_block<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate); \
template __global__ void gather_block_random<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n); \
template __global__ void scatter_block_random<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n); \
template __global__ void sg_block<V>(double* source, double* target, sgIdx_t* pat_gath, sgIdx_t* pat_scat, spSize_t pat_len, size_t delta_gather, size_t delta_scatter, int wpt, char validate); \
template __global__ void multiscatter_block<V>(double* source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate); \
template __global__ void multigather_block<V>(double* source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate);

INSTANTIATE(1);
INSTANTIATE(2);
INSTANTIATE(4);
INSTANTIATE(5);
INSTANTIATE(8);
INSTANTIATE(16);
INSTANTIATE(32);
INSTANTIATE(64);
INSTANTIATE(128);
INSTANTIATE(256);
INSTANTIATE(512);
INSTANTIATE(1024);
INSTANTIATE(2048);
INSTANTIATE(4096);
