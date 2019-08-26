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

//Small index buffer version of gather
void gather_smallbuf_serial(
        sgData_t** restrict target, 
        sgData_t* const restrict source, 
        sgIdx_t* const restrict pat, 
        size_t pat_len, 
        size_t delta, 
        size_t n, 
        size_t target_len) {
        
	for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + delta * i; 
           //Pick which 8 elements are written to in a way that 
	   //is hard to optimize out with a compiler
	   sgData_t *tl = target[0] + pat_len*(i%target_len);
       
	   #pragma novector    
	   for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
}


void scatter_smallbuf_serial(
        sgData_t* restrict target, 
        sgData_t** const restrict source, 
        sgIdx_t* const restrict pat, 
        size_t pat_len, 
        size_t delta, 
        size_t n, 
        size_t source_len) {
        
	for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i; 
           sgData_t *sl = source[0] + pat_len*(i%source_len);
          
	   #pragma novector 
	   for (size_t j = 0; j < pat_len; j++) {
               tl[pat[j]] = sl[j];
           }
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
