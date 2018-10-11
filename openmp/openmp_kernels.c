#include "openmp_kernels.h"

#define SIMD 4

void sg_omp(
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

  if(B == 1){
#pragma omp parallel for simd safelen(SIMD)
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[sir[i]+b];
        }
	}
  }
}

void scatter_omp(
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

  if(B == 1){
#pragma omp parallel for simd safelen(SIMD)
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[i];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[i+b];
        }
	}
  }

}

void gather_omp(
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


  if(B == 1){
#pragma omp parallel for simd safelen(SIMD)
	for(long i = 0; i < n; i++){
	    tr[i] = sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        tr[i+b] = sr[sir[i]+b];
        }
	}
  }

}

void sg_accum_omp(
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


  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[sir[i]+b];
        }
	}
  }
}

void scatter_accum_omp(
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

  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[i];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[i+b];
        }
	}
  }
}

void gather_accum_omp(
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


  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    tr[i] += sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[i+b] += sr[sir[i]+b];
        }
	}
  }
}
