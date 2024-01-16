#ifndef SYCL_BACKEND_HPP
#define SYCL_BACKEND_HPP
#include <sycl/sycl.hpp>
#include <stdint.h>
extern "C" {
#include "../include/parse-args.h"
#include "sgbuf.h"
}

extern void my_kernel_wrapper(unsigned int dim, unsigned int* grid, unsigned int* block);

extern float sycl_sg_wrapper(enum sg_kernel kernel,
		      size_t vector_len,
		      long unsigned dim, long unsigned* grid, long unsigned* block,
		      double* target, double *source,
		      long* ti, long* si, unsigned int shmem, sycl::queue* q);

extern float sycl_block_multiscatter_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q);
extern float sycl_block_multigather_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q);
extern float sycl_block_sg_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* pat_gath_dev,
        sgIdx_t* pat_scat_dev,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q);
extern float sycl_block_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt, size_t morton, uint32_t *order, uint32_t *order_dev, int stride,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q);
extern float sycl_block_random_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt, size_t seed, sycl::queue* q);
extern float sycl_new_wrapper(long unsigned dim, long unsigned* grid, long unsigned* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        sgIdx_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap, int wpt, sycl::queue* q);

void create_dev_buffers_sycl(sgDataBuf *source, sycl::queue* que)
{
    (void*)source->dev_ptr_cuda = (void*)sycl::malloc_device(source->size, *que);
    if (source->dev_ptr_cuda == nullptr) {
        printf("Could not allocate gpu memory (%zu bytes)\n", source->size);
        exit(1);
    }
}

#endif
