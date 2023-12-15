#ifndef CUDA_BACKEND_HH
#define CUDA_BACKEND_HH

#include <cstddef>

float cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const int pattern_length, const size_t count,
    const size_t wrap);
float cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const int pattern_length, const size_t count,
    const size_t wrap);
float cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const int pattern_length, const size_t count,
    const size_t wrap);
float cuda_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const int pattern_length, const size_t count, const size_t wrap);
float cuda_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const int pattern_length, const size_t count, const size_t wrap);
#endif
