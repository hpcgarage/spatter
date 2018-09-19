#include <stdio.h>
#include <cuda.h>
#include "sgbuf.h"
#include "cuda-backend.h"

void create_dev_buffers_cuda(sgDataBuf* source, sgDataBuf* target, 
                            sgIndexBuf* si, sgIndexBuf *ti, 
                            size_t block_len){
    cudaMalloc((void **)&(source->dev_ptr), source->size);
    cudaMalloc((void **)&(target->dev_ptr), target->size);
    cudaMalloc((void **)&(si->dev_ptr), si->size);
    cudaMalloc((void **)&(ti->dev_ptr), ti->size);
}
