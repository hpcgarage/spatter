#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H
#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/parse-args.h"
#include "sgbuf.h"

extern void my_kernel_wrapper(unsigned int dim, unsigned int* grid, unsigned int* block);

extern float cuda_sg_wrapper(enum sg_kernel kernel,
                       size_t vector_len,
                       long unsigned dim, long unsigned* grid, long unsigned* block,
                       double* target, double *source,
                       long* ti, long* si, unsigned int shmem);

extern float cuda_block_multiscatter_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate);
extern float cuda_block_multigather_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate);
extern float cuda_block_sg_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* pat_gath_dev,
        sgIdx_t* pat_scat_dev,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate);
extern float cuda_block_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        double *target,
        sgIdx_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt, size_t morton, uint32_t *order, uint32_t *order_dev, int stride,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate);
extern float cuda_block_random_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt, size_t seed);
extern float cuda_new_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        sgIdx_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt);

void create_dev_buffers_cuda(sgDataBuf *source);

int find_device_cuda(char *name);
#endif
