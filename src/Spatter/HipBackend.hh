#ifndef HIP_BACKEND_HH
#define HIP_BACKEND_HH

#include <cstddef>

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
            hipGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

float hip_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count);
float hip_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count);
float hip_scatter_atomic_wrapper(const size_t *pattern, double *sparse,
    const double *dense, size_t pattern_length, size_t delta,
    size_t wrap, size_t count);
float hip_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count);
float hip_scatter_gather_atomic_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, size_t pattern_length,
    size_t delta_scatter, size_t delta_gather, size_t wrap,
    size_t count);
float hip_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    size_t pattern_length, size_t delta, size_t wrap,
    size_t count);
float hip_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap,
    size_t count);
float hip_multi_scatter_atomic_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    size_t pattern_length, size_t delta, size_t wrap,
    size_t count);

#endif
