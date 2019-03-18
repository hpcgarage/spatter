#ifndef SERIAL_KERNELS_H
#define SERIAL_KERNELS_H

#include <stdlib.h>
#include "../include/sgtype.h"

void sg_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
void scatter_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
void gather_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
void sg_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
void scatter_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
void gather_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi);
#endif
