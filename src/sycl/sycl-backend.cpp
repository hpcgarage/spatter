#ifndef _GNU_SOURCE
    #define _GNU_SOURCE //needed for string.h to include strcasestr
#endif
#include <stdio.h>
#include <string.h>
#include "sgbuf.h"
#include "sycl-backend.hpp"

void create_dev_buffers_cuda(sgDataBuf* source, sycl::queue* q)
{
    source->dev_ptr_cuda = sycl::malloc_device(source->size, *q);
    if (source->dev_ptr_cuda != nullptr) {
        printf("Could not allocate gpu memory (%zu bytes)\n", source->size);
        exit(1);
    }
}
