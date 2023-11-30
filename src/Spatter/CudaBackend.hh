#ifndef CUDA_BACKEND_HH
#define CUDA_BACKEND_HH

void cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const int pattern_length);
void cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const int pattern_length);

#endif
