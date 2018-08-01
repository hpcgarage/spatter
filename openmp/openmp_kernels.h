#ifndef OMP_KERNELS_H
#define OMP_KERNELS_H

#include <omp.h>
#include <stdlib.h>
#include "../include/sgtype.h"

void sg_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
void scatter_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
void gather_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
void sg_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
void scatter_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
void gather_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B);
#endif
