#include "sgtype.h"
#include "sgbuf.h"
#include "mt64.h"

void random_data(sgData_t *buf, size_t len){
    for(size_t i = 0; i < len; i++){
        buf[i] = rand() % 10; 
    }
}

void linear_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t stride, int randomize){
    sgIdx_t *idx_cur = idx;
    for(size_t j = 0; j < worksets; j++){
        for(size_t i = 0; i < len; i++){
            idx_cur[i] = i * stride;
        }
        idx_cur = idx_cur + len;
    }
    
    if(randomize){
        unsigned long long init[4] = {0x12345ULL, 0x23456ULL, 
                                      0x34567ULL,0x45678ULL};
        int length = 4;
        init_by_array64(init, length);

        for(size_t i = 0; i < len-2; i++){
            size_t j = (genrand64_int64() % (len-i)) + i; 
            for(size_t k = 0; k < worksets; k++) {
                size_t tmp = idx[k*len+i];
                idx[k*len+i] = idx[k*len+j];
                idx[k*len+j] = tmp;
            }
        }
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
