#include "openmp_kernels.h"

#define SIMD 4

void sg_omp(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n,
            long B)
{
  if(B == 1){
#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[si[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        target[ti[i]+b] = source[si[i]+b];
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
            long B)
{

  if(B == 1){
#pragma omp parallel for simd safelen(SIMD)
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[i];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        target[ti[i]+b] = source[i+b];
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
            long B)
{

  if(B == 1){
//Users may want to set a specific safelen value like 32
#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
	for(long i = 0; i < n; i++){
	    target[i] = source[si[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	#pragma omp simd safelen(SIMD)
        for(int b = 0; b < B; b++){
	        target[i+b] = source[si[i]+b];
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
            long B)
{

  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[si[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        target[ti[i]+b] += source[si[i]+b];
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
            long B)
{
  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[i];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        target[ti[i]+b] += source[i+b];
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
            long B)
{

  if(B == 1){
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[i] += source[si[i]];
	}
  }
  else{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        target[i+b] += source[si[i]+b];
        }
	}
  }
}
