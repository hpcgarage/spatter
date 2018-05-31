#ifndef SGBUF_H
#define SGBUF_H

#include "sgtype.h"
#include "cl-helper.h"

typedef struct sgDataBuf_t{
    SGTYPE *host_ptr;
    cl_mem dev_ptr;
    size_t len;
    size_t size;
    size_t block_len;
}sgDataBuf;

typedef struct sgIndexBuf_t{
    cl_ulong *host_ptr;
    cl_mem dev_ptr;
    size_t len;
    size_t size;
}sgIndexBuf;


void random_data(SGTYPE *buf, size_t len);
void linear_indices(cl_ulong *idx, size_t len, size_t worksets);
cl_mem clCreateBufferSafe(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr);

#endif
