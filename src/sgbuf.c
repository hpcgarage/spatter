#include <stdio.h>
#include <stdlib.h>
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
    
    // Fisher-Yates Shuffling
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

void wrap_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t stride, size_t wrap) {
    if(wrap > stride || stride == 1){
        linear_indices(idx, len, worksets, stride, 0);
        return;
    }
    sgIdx_t *idx_cur = idx;
    for(size_t j = 0; j < worksets; j++) {
        for(size_t w = 0; w < wrap; w++){
            size_t offset = (stride-(w*(stride/wrap))-1);
            for(size_t i = 0; i < len/wrap; i++){
                idx_cur[i + (len/wrap)*w] = offset + stride*i;
            }
        }
        idx_cur = idx_cur + len;
    }
}

//Mostly Stride-1
void ms1_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t run, size_t gap){
    sgIdx_t *idx_cur = idx;
    for(size_t j = 0; j < worksets; j++){
        for(size_t i = 0; i < len; i++){

            idx_cur[i] = (i / run) * gap + (i % run);
        }
        idx_cur = idx_cur + len;
    }

}

struct instruction get_random_instr (struct trace tr) {
    double r = (double)rand() / (double)RAND_MAX;
    for (int i = 0; i < tr.length-1; i++) {
        if (tr.in[i].cpct > r) {
            return tr.in[i];
        }
    }
    return tr.in[tr.length-1];
}

//returns the size of the buffer required
size_t trace_indices( sgIdx_t *idx, size_t len, struct trace tr) {
//for now, assume that all specified numbers are for 8-byte data types
// and reads are 8 byte alignd
    size_t data_type_size = 8;
    size_t cur = 0; 
    int done = 0;
    while (cur < len && !done) {
        struct instruction in = get_random_instr (tr);
        int i;
        for (i = 0; i < in.length ; i++) {
            if (i + cur < len) {
                idx[i+cur] = in.delta[i];
            } else {
                done = 1;
                break;
            }
        }
        cur += i;
    }
    assert (cur == len);

    size_t min = idx[0] / 8;
    for (size_t i = 1; i < len; i++) {
        idx[i] = idx[i-1] + idx[i] / 8;
        if (idx[i] < min) 
            min = idx[i];
    }

    size_t max = idx[0];
    for (size_t i = 0; i < len; i++) {
        idx[i] = idx[i] - min;
        if (idx[i] > max) 
            max = idx[i];
    }

    return max;
}
#ifdef USE_OPENCL
cl_mem clCreateBufferSafe(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr){
    cl_int err;
    cl_mem buf = clCreateBuffer(context, flags, size, host_ptr, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer");
    return buf;
}
#endif
