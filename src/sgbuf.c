#include "sgtype.h"
#include "sgbuf.h"

void random_data(SGTYPE_C *buf, size_t len){
    for(size_t i = 0; i < len; i++){
        buf[i] = rand() % 10; 
    }
}

void linear_indices(cl_ulong *idx, size_t len, size_t worksets){
    cl_ulong *idx_cur = idx;
    for(size_t j = 0; j < worksets; j++){
        for(size_t i = 0; i < len; i++){
            idx_cur[i] = i;
        }
        idx_cur = idx_cur + len;
    }
}

void random_indices(cl_ulong *idx, size_t len, size_t worksets){
    cl_ulong *idx_cur = idx;
    for(size_t j = 0; j < worksets; j++){
        for(size_t i = 0; i < len; i++){
            idx_cur[i] = rand()%len;
        }
        idx_cur = idx_cur + len;
    }
}

#ifdef USE_OPENCL
cl_mem clCreateBufferSafe(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr){
    cl_int err;
    cl_mem buf = clCreateBuffer(context, flags, size, host_ptr, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer");
    return buf;
}
#endif
