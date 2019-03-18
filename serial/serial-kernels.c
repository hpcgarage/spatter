#include "serial_kernels.h"

void sg_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *tir = ti     + oi; 
  long     *sir = si     + oi; 

	#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[sir[i]];
	}
}

void scatter_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *tir = ti     + oi; 

	#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[i];
	}

}

void gather_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *sir = si     + oi; 


	#pragma novector
	for(long i = 0; i < n; i++){
	    tr[i] = sr[sir[i]];
	}
}

void sg_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *tir = ti     + oi; 
  long     *sir = si     + oi; 


	#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[sir[i]];
	}
}

void scatter_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *tir = ti     + oi; 

#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[i];
	}
}

void gather_accum_serial(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  sgData_t *tr  = target + ot;
  sgData_t *sr  = source + os;
  long     *sir = si     + oi; 


	#pragma novector
	for(long i = 0; i < n; i++){
	    tr[i] += sr[sir[i]];
	}
}
