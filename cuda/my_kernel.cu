#include <stdio.h>
#include "cuda_kernels.h"
#include "../include/parse-args.h"

#define typedef uint unsigned int
__global__ void my_kernel(){
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void scatter1(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir  = ti     + oi;
    tr[tir[gid]] = sr[gid];
}

__global__ void scatter2(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir =  ti     + oi;
    tr[tir[2*gid]] = sr[2*gid];
    tr[tir[2*gid+1]] = sr[2*gid+1];
}


__global__ void scatter4(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir =  ti     + oi;
    tr[tir[4*gid]] = sr[4*gid];
    tr[tir[4*gid+1]] = sr[4*gid+1];
    tr[tir[4*gid+2]] = sr[4*gid+2];
    tr[tir[4*gid+3]] = sr[4*gid+3];
}

__global__ void scatter8(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir =  ti     + oi;
    tr[tir[8*gid+0]] = sr[8*gid+0];
    tr[tir[8*gid+1]] = sr[8*gid+1];
    tr[tir[8*gid+2]] = sr[8*gid+2];
    tr[tir[8*gid+3]] = sr[8*gid+3];
    tr[tir[8*gid+4]] = sr[8*gid+4];
    tr[tir[8*gid+5]] = sr[8*gid+5];
    tr[tir[8*gid+6]] = sr[8*gid+6];
    tr[tir[8*gid+7]] = sr[8*gid+7];
}

__global__ void scatter16(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir =  ti     + oi;
    tr[tir[16*gid+0]] = sr[16*gid+0];
    tr[tir[16*gid+1]] = sr[16*gid+1];
    tr[tir[16*gid+2]] = sr[16*gid+2];
    tr[tir[16*gid+3]] = sr[16*gid+3];
    tr[tir[16*gid+4]] = sr[16*gid+4];
    tr[tir[16*gid+5]] = sr[16*gid+5];
    tr[tir[16*gid+6]] = sr[16*gid+6];
    tr[tir[16*gid+7]] = sr[16*gid+7];
    tr[tir[16*gid+8]] = sr[16*gid+8];
    tr[tir[16*gid+9]] = sr[16*gid+9];
    tr[tir[16*gid+10]] = sr[16*gid+10];
    tr[tir[16*gid+11]] = sr[16*gid+11];
    tr[tir[16*gid+12]] = sr[16*gid+12];
    tr[tir[16*gid+13]] = sr[16*gid+13];
    tr[tir[16*gid+14]] = sr[16*gid+14];
    tr[tir[16*gid+15]] = sr[16*gid+15];
}

__global__ void gather1(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    tr[gid] = sr[sir[gid]];
}

__global__ void gather2(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    tr[2*gid+0] = sr[sir[2*gid+0]];
    tr[2*gid+1] = sr[sir[2*gid+1]];
}

__global__ void gather4(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    tr[4*gid+0] = sr[sir[4*gid+0]];
    tr[4*gid+1] = sr[sir[4*gid+1]];
    tr[4*gid+2] = sr[sir[4*gid+2]];
    tr[4*gid+3] = sr[sir[4*gid+3]];
}

__global__ void gather8(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    tr[8*gid+0] = sr[sir[8*gid+0]];
    tr[8*gid+1] = sr[sir[8*gid+1]];
    tr[8*gid+2] = sr[sir[8*gid+2]];
    tr[8*gid+3] = sr[sir[8*gid+3]];
    tr[8*gid+4] = sr[sir[8*gid+4]];
    tr[8*gid+5] = sr[sir[8*gid+5]];
    tr[8*gid+6] = sr[sir[8*gid+6]];
    tr[8*gid+7] = sr[sir[8*gid+7]];
}

__global__ void gather16(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    double temp[16];
    
    temp[0] = sr[sir[16*gid+0]];
    temp[1] = sr[sir[16*gid+1]];
    temp[2] = sr[sir[16*gid+2]];
    temp[3] = sr[sir[16*gid+3]];
    temp[4] = sr[sir[16*gid+4]];
    temp[5] = sr[sir[16*gid+5]];
    temp[6] = sr[sir[16*gid+6]];
    temp[7] = sr[sir[16*gid+7]];
    temp[8] = sr[sir[16*gid+8]];
    temp[9] = sr[sir[16*gid+9]];
    temp[10] = sr[sir[16*gid+10]];
    temp[11] = sr[sir[16*gid+11]];
    temp[12] = sr[sir[16*gid+12]];
    temp[13] = sr[sir[16*gid+13]];
    temp[14] = sr[sir[16*gid+14]];
    temp[15] = sr[sir[16*gid+15]];

    memcpy(tr+gid*16, temp, sizeof(double)*16);
}

/*
__global__ void gather16(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* sir  = si     + oi;
    tr[16*gid+0] = sr[sir[16*gid+0]];
    tr[16*gid+1] = sr[sir[16*gid+1]];
    tr[16*gid+2] = sr[sir[16*gid+2]];
    tr[16*gid+3] = sr[sir[16*gid+3]];
    tr[16*gid+4] = sr[sir[16*gid+4]];
    tr[16*gid+5] = sr[sir[16*gid+5]];
    tr[16*gid+6] = sr[sir[16*gid+6]];
    tr[16*gid+7] = sr[sir[16*gid+7]];
    tr[16*gid+8] = sr[sir[16*gid+8]];
    tr[16*gid+9] = sr[sir[16*gid+9]];
    tr[16*gid+10] = sr[sir[16*gid+10]];
    tr[16*gid+11] = sr[sir[16*gid+11]];
    tr[16*gid+12] = sr[sir[16*gid+12]];
    tr[16*gid+13] = sr[sir[16*gid+13]];
    tr[16*gid+14] = sr[sir[16*gid+14]];
    tr[16*gid+15] = sr[sir[16*gid+15]];

}*/
__global__ void sg1(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;
    tr[tir[gid]] = sr[sir[gid]];
}

__global__ void sg2(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;
    tr[tir[2*gid+0]] = sr[sir[2*gid+0]];
    tr[tir[2*gid+1]] = sr[sir[2*gid+1]];
}

__global__ void sg4(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;
    tr[tir[4*gid+0]] = sr[sir[4*gid+0]];
    tr[tir[4*gid+1]] = sr[sir[4*gid+1]];
    tr[tir[4*gid+2]] = sr[sir[4*gid+2]];
    tr[tir[4*gid+3]] = sr[sir[4*gid+3]];
}
__global__ void sg8(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;
    tr[tir[8*gid+0]] = sr[sir[8*gid+0]];
    tr[tir[8*gid+1]] = sr[sir[8*gid+1]];
    tr[tir[8*gid+2]] = sr[sir[8*gid+2]];
    tr[tir[8*gid+3]] = sr[sir[8*gid+3]];
    tr[tir[8*gid+4]] = sr[sir[8*gid+4]];
    tr[tir[8*gid+5]] = sr[sir[8*gid+5]];
    tr[tir[8*gid+6]] = sr[sir[8*gid+6]];
    tr[tir[8*gid+7]] = sr[sir[8*gid+7]];
}

__global__ void sg16(double* target, 
                    double* source, 
                    long* ti, 
                    long* si, 
                    long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long* tir  = ti     + oi;
    long* sir  = si     + oi;
    tr[tir[16*gid+0]] = sr[sir[16*gid+0]];
    tr[tir[16*gid+1]] = sr[sir[16*gid+1]];
    tr[tir[16*gid+2]] = sr[sir[16*gid+2]];
    tr[tir[16*gid+3]] = sr[sir[16*gid+3]];
    tr[tir[16*gid+4]] = sr[sir[16*gid+4]];
    tr[tir[16*gid+5]] = sr[sir[16*gid+5]];
    tr[tir[16*gid+6]] = sr[sir[16*gid+6]];
    tr[tir[16*gid+7]] = sr[sir[16*gid+7]];
    tr[tir[16*gid+8]] = sr[sir[16*gid+8]];
    tr[tir[16*gid+9]] = sr[sir[16*gid+9]];
    tr[tir[16*gid+10]] = sr[sir[16*gid+10]];
    tr[tir[16*gid+11]] = sr[sir[16*gid+11]];
    tr[tir[16*gid+12]] = sr[sir[16*gid+12]];
    tr[tir[16*gid+13]] = sr[sir[16*gid+13]];
    tr[tir[16*gid+14]] = sr[sir[16*gid+14]];
    tr[tir[16*gid+15]] = sr[sir[16*gid+15]];
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

extern "C" void my_kernel_wrapper(uint dim, uint* grid, uint* block){
    dim3 grid_dim, block_dim;
    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return;

    my_kernel<<<grid_dim,block_dim>>>();
    cudaDeviceSynchronize();
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
            scatter1<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            scatter2<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            scatter4<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            scatter8<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            scatter16<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else 
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GATHER)
    {
        if (vector_len == 1)
            gather1<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            gather2<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            gather4<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            gather8<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            gather16<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else 
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == SG)
    {
        if (vector_len == 1)
            sg1<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 2)
            sg2<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 4)
            sg4<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 8)
            sg8<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
        else if (vector_len == 16)
            sg16<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);
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
