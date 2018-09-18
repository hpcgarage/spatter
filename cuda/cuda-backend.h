#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H
#include <cuda_runtime.h>
extern void my_kernel_wrapper(unsigned int dim, unsigned int* grid, unsigned int* block);

extern void scatter_wrapper(uint dim, uint* grid, uint* block, 
                            double* target, double *source, 
                            long* ti, long* si, 
                            long ot, long os, long oi);

void create_dev_buffers_cuda(sgDataBuf *source, sgDataBuf *targt, 
                             sgIndexBuf *si, sgIndexBuf *ti, 
                             size_t block_len);
#endif
