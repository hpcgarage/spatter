#ifndef OMP_KERNELS_H
#define OMP_KERNELS_H

#if defined( USE_OPENMP )
#include <omp.h>
#endif
#include <stdlib.h>
#include <stdint.h>
#include "../include/sgtype.h"

void sg_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);
void scatter_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);
void gather_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);
void sg_accum_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);
void scatter_accum_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);
void gather_accum_omp(
            sgData_t* restrict target,
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n);

void gather(
        sgData_t* restrict target,
        sgData_t* restrict source,
        sgIdx_t*  restrict pat,
        size_t    pat_len,
        size_t    delta,
        size_t    n);
void scatter(
        sgData_t* restrict target,
        sgData_t* restrict source,
        sgIdx_t*  restrict pat,
        size_t    pat_len,
        size_t    delta,
        size_t    n);
void gather_stride_os(
        sgData_t* restrict target,
        sgData_t* restrict source,
        sgIdx_t*  restrict pat,
        size_t    pat_len,
        size_t    delta,
        size_t    n,
        size_t    target_wrap);
void gather_stride8(
        sgData_t* restrict target,
        sgData_t* restrict source,
        sgIdx_t*  restrict pat,
        size_t    delta,
        size_t    n);
void gather_stride16(
        sgData_t* restrict target,
        sgData_t* restrict source,
        sgIdx_t*  restrict pat,
        size_t    stride,
        size_t    delta,
        size_t    n);

void multigather_smallbuf(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len);

void multigather_smallbuf_morton(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        uint32_t *order);

void multiscatter_smallbuf(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len);

void multigather_smallbuf_random(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        long initstate);

void multiscatter_smallbuf_random(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len,
        long initstate);

void multigather_smallbuf_multidelta(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t *delta,
        size_t n,
        size_t target_len,
        size_t delta_len);


void sg_smallbuf(
        sgData_t* restrict gather,
        sgData_t* restrict scatter,
        ssize_t* const restrict gather_pat,
        ssize_t* const restrict scatter_pat,
        size_t pat_len,
        size_t delta_gather,
        size_t delta_scatter,
        size_t n,
        size_t wrap);

void gather_smallbuf(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len);

void gather_smallbuf_morton(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        uint32_t *order);

void scatter_smallbuf(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len);

void gather_smallbuf_random(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        long initstate);

void scatter_smallbuf_random(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len,
        long initstate);

void gather_smallbuf_multidelta(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t *delta,
        size_t n,
        size_t target_len,
        size_t delta_len);



#endif
