#include <stdio.h>
//#include "cuda-backend.h"
#include "cuda_kernels.h"

#define typedef uint unsigned int
__global__ void my_kernel(){
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void scatter(double* target, 
                        double* source, 
                        long* ti, 
                        long* si, 
                        long ot, long os, long oi)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double* tr = target + ot;
    double* sr = source + os;
    long *tir = ti;
    tr[tir[gid]] = sr[gid];
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

extern "C" void scatter_wrapper(uint dim, uint* grid, uint* block, 
                                double* target, double *source, 
                                long* ti, long* si, 
                                long ot, long os, long oi){
    dim3 grid_dim, block_dim;
    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return;
    scatter<<<grid_dim,block_dim>>>(target, source, ti, si, ot, os, oi);

}
