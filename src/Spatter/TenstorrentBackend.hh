#ifndef TENSTORRENT_BACKEND_HH
#define TENSTORRENT_BACKEND_HH

#include <cstddef>

// tt-metal has no raw device pointers, so these return an opaque handle that
// Configuration<Tenstorrent> keeps in the pointer slots ConfigurationBase
// provides. Nothing on the host dereferences them. page_size is the DRAM page:
// sizeof(double) for sparse/dense, pattern_length*sizeof(uint32_t) for patterns.
void *tt_device_alloc(size_t bytes, size_t page_size);
void tt_device_free(void *handle);
void tt_memcpy_h2d(void *handle, const void *src, size_t bytes);
void tt_memcpy_d2h(void *dst, const void *handle, size_t bytes);

// Narrows Spatter's size_t pattern to the uint32 the device kernel uses.
void tt_pattern_upload(void *handle, const size_t *pattern, size_t length);

// Mirrors CudaBackend.hh. Return value is elapsed milliseconds.
// multi_* and *_atomic are declared but return a negative sentinel.

float tt_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count);

float tt_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count);

float tt_scatter_atomic_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count);

float tt_gather_scatter_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count);

float tt_gather_scatter_atomic_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count);

float tt_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count);

float tt_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count);

float tt_multi_scatter_atomic_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count);

#endif
