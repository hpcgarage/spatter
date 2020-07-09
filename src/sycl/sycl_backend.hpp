#ifndef SYCL_BACKEND

#include "../../include/sgtype.h"

double sycl_gather(double* src, size_t src_size, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned long global_work_size, unsigned long local_work_size);

#endif