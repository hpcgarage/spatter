#include "serial-kernels.h"

void sg_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
	#pragma novector
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[si[i]];
	}
}

void scatter_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
	#pragma novector
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[i];
	}

}

void gather_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
	#pragma novector
	for(long i = 0; i < n; i++){
	    target[i] = source[si[i]];
	}
}

void sg_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
	#pragma novector
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[si[i]];
	}
}

void scatter_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
	#pragma novector
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[i];
	}
}

void gather_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{

	#pragma novector
	for(long i = 0; i < n; i++){
	    target[i] += source[si[i]];
	}
}
