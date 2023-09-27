#include <stdio.h>
#include <string.h>
#include "sgbuf.h"
#include "sycl-backend.hpp"

void create_dev_buffers_sycl(sgDataBuf* source, sycl::queue* q)
{
    (void*)source->dev_ptr_cuda = (void*)sycl::malloc_device(source->size, *q);
    if (source->dev_ptr_cuda != nullptr) {
        printf("Could not allocate gpu memory (%zu bytes)\n", source->size);
        exit(1);
    }
}
