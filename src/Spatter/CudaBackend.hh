#ifndef CUDA_BACKEND_HH
#define CUDA_BACKEND_HH

#include <cstddef>

void cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const int pattern_length);
void cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const int pattern_length);
void cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const int pattern_length);
void cuda_multi_gather(const size_t *pattern, const size_t *pattern_gather,
    const double *sparse, double *dense, const int pattern_length);
void cuda_multi_scatter(const size_t *pattern, const size_t *pattern_scatter,
    double *sparse, const double *dense, const int pattern_length);
#endif
