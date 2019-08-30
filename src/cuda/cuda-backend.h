#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H
#include <cuda_runtime.h>
#include "../include/parse-args.h"
#include "sgbuf.h"

extern void my_kernel_wrapper(unsigned int dim, unsigned int* grid, unsigned int* block);

extern float cuda_sg_wrapper(enum sg_kernel kernel, 
                       size_t vector_len, 
                       uint dim, uint* grid, uint* block, 
                       double* target, double *source, 
                       long* ti, long* si, unsigned int shmem);

extern float cuda_block_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        sgIdx_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt);
extern float cuda_new_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        sgIdx_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt);

void create_dev_buffers_cuda(sgDataBuf *source);

int find_device_cuda(char *name);
#endif
