#ifndef SERIAL_KERNELS_H
#define SERIAL_KERNELS_H

#include <stdlib.h>
#include "../include/sgtype.h"

void sg_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n);

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

void sg_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n);
void scatter_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n);
void gather_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n);
#endif
