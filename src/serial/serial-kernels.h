#ifndef SERIAL_KERNELS_H
#define SERIAL_KERNELS_H

#include <stdlib.h>
#include "../include/sgtype.h"

void multigather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len);

void multiscatter_smallbuf_serial(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len);

void gather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len);

void scatter_smallbuf_serial(
        sgData_t* restrict target,
        sgData_t** restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len);

void sg_smallbuf_serial(
        sgData_t* restrict gather,
        sgData_t* restrict scatter,
        ssize_t* const restrict gather_pat,
        ssize_t* const restrict scatter_pat,
        size_t pat_len,
        size_t delta_gather,
        size_t delta_scatter,
        size_t n,
        size_t wrap);
#endif
