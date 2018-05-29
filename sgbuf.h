#ifndef SGBUF
#define SGBUF

#include "sgtype.h"
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

#endif
