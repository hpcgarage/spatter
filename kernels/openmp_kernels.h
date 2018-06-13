#ifndef OMP_KERNELS_H
#define OMP_KERNELS_H
#include "../sgtype.h"
void sg_omp(SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
void scatter_omp(SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
void gather_omp(SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
#endif
