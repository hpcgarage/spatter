#define _GNU_SOURCE //needed for string.h to include strcasestr
#include <stdio.h>
#include <string.h>
#include "sgbuf.h"
#include "cuda-backend.h"

#define cudaSilent(a) if(a!=cudaSuccess) exit(0);

void create_dev_buffers_cuda(sgDataBuf* source, sgDataBuf* target, 
                            sgIndexBuf* si, sgIndexBuf *ti)
{
    cudaError_t ret;
    ret = cudaMalloc((void **)&(source->dev_ptr_cuda), source->size); cudaSilent(ret);
    ret = cudaMalloc((void **)&(target->dev_ptr_cuda), target->size); cudaSilent(ret);
    ret = cudaMalloc((void **)&(si->dev_ptr_cuda), si->size);         cudaSilent(ret);
    ret = cudaMalloc((void **)&(ti->dev_ptr_cuda), ti->size);         cudaSilent(ret);
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
