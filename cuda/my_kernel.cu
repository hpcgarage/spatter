#include <stdio.h>
#include "cuda_kernels.h"
#include "../include/parse-args.h"

#define typedef uint unsigned int

template<int v>
__global__ void scatter_t(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);
    double* tr = target + ot;
    double* sr = source + os;
    long *tir =  ti     + oi;

    double buf[v];
    long idx[v]; 

    for(int i = 0; i < v; i++){
        buf[i] = sr[gid+i];    
    }

    for(int i = 0; i < v; i++){
       idx[i] = tir[gid+i]; 
    }

    for(int i = 0; i < v; i++){
        tr[idx[i]] = buf[i];
    }
}


template<int v>
__global__ void gather_t(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    double buf[v];

    for(int i = 0; i < v; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < v; i++){
        tr[gid+i] = buf[i];
    }

}

template<int v>
__global__ void sg_t(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = v*(blockIdx.x * blockDim.x + threadIdx.x);
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;

    long sidx[v];
    long tidx[v];

    for(int i = 0; i < v; i++){
        sidx[i] = sir[gid+i];
    }
    for(int i = 0; i < v; i++){
        tidx[i] = tir[gid+i];
    }
    for(int i = 0; i < v; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }

}
#define INSTANTIATE(V)\
template __global__ void scatter_t<V>(double* target, double* source, long* ti, long* si, long ot, long os, long oi);\
template __global__ void gather_t<V>(double* target, double* source, long* ti, long* si, long ot, long os, long oi); \
template __global__ void sg_t<V>(double* target, double* source, long* ti, long* si, long ot, long os, long oi);
INSTANTIATE(1);
INSTANTIATE(2);
INSTANTIATE(4);
INSTANTIATE(5);
INSTANTIATE(8);
INSTANTIATE(16);
INSTANTIATE(32);
INSTANTIATE(64);

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
                                size_t block_len, 
                                size_t vector_len, 
                                uint dim, uint* grid, uint* block, 
                                double* target, double *source, 
                                long* ti, long* si, 
                                long ot, long os, long oi){
    dim3 grid_dim, block_dim;
    cudaEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    if(kernel == SCATTER)
    {
        if (vector_len == 1)
            scatter_t<1><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            scatter_t<2><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            scatter_t<4><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 5)
            scatter_t<5><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            scatter_t<8><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            scatter_t<16><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 32)
            scatter_t<32><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 64)
            scatter_t<64><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else 
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GATHER)
    {
        if (vector_len == 1)
            scatter_t<1><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            scatter_t<2><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            scatter_t<4><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 5)
            scatter_t<5><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            scatter_t<8><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            scatter_t<16><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 32)
            scatter_t<32><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 64)
            scatter_t<64><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else 
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == SG)
    {
        if (vector_len == 1)
            sg_t<1><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            sg_t<2><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            sg_t<4><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 5)
            sg_t<5><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            sg_t<8><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            sg_t<16><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 32)
            sg_t<32><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 64)
            sg_t<64><<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    return time_ms;

}
