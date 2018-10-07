#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H
#include <cuda_runtime.h>
#include "../include/parse-args.h"
#include "sgbuf.h"

extern void my_kernel_wrapper(unsigned int dim, unsigned int* grid, unsigned int* block);

extern float cuda_sg_wrapper(enum sg_kernel kernel, size_t block_len, 
                       size_t vector_len, 
                       uint dim, uint* grid, uint* block, 
                       double* target, double *source, 
                       long* ti, long* si, 
                       long ot, long os, long oi, unsigned int shmem);

void create_dev_buffers_cuda(sgDataBuf *source, sgDataBuf *targt, 
                             sgIndexBuf *si, sgIndexBuf *ti, 
                             size_t block_len);

int find_device_cuda(char *name);
#endif
