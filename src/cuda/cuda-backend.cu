#ifndef _GNU_SOURCE
    #define _GNU_SOURCE //needed for string.h to include strcasestr
#endif
#include <stdio.h>
#include <string.h>
#include "sgbuf.h"
#include "cuda-backend.h"

#define cudaSilent(a) if(a!=cudaSuccess) exit(0);

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
