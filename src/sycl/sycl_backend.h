#ifndef SYCL_BACKEND
#define SYCL_BACKEND

#include "../../include/sgtype.h"

extern double sycl_gather(double* src, size_t src_size, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned int* grid, unsigned int* block, unsigned int dim);
extern double sycl_scatter(double* src, size_t src_size, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned int* grid, unsigned int* block, unsigned int dim);

#endif
