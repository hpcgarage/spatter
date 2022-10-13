#ifndef SERIAL_KERNELS_H
#define SERIAL_KERNELS_H

#include <stdlib.h>
#include "../include/sgtype.h"

void gather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len);

void scatter_smallbuf_serial(
        sgData_t* restrict target,
        sgData_t** restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len);

void sg_smallbuf_serial(
        sgData_t* restrict gather,
        sgData_t* restrict scatter,
        sgIdx_t* const restrict gather_pat,
        sgIdx_t* const restrict scatter_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap);
#endif
