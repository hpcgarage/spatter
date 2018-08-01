#include "openmp_kernels.h"

void sg_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *tir = ti     + oi; 
  long     *sir = si     + oi; 


  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[sir[i]+b];
        }
	}
  }
}

void scatter_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *tir = ti     + oi; 

  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[i];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[i+b];
        }
	}
  }
}

void gather_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *sir = si     + oi; 


  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[i] = sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[i+b] = sr[sir[i]+b];
        }
	}
  }
}
void sg_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *tir = ti     + oi; 
  long     *sir = si     + oi; 


  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[sir[i]+b];
        }
	}
  }
}

void scatter_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *tir = ti     + oi; 

  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[i];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[i+b];
        }
	}
  }
}

void gather_accum_omp(
            SGTYPE_C* restrict target, 
            long*     restrict ti,
            SGTYPE_C* restrict source,
            long*     restrict si,
            size_t n,
            long ot, 
            long os, 
            long oi, 
            long B)
{
  SGTYPE_C *tr  = target + ot;
  SGTYPE_C *sr  = source + os;
  long     *sir = si     + oi; 


  if(B == 1){
#pragma omp parallel for
	for(long i = 0; i < n; i++){
	    tr[i] += sr[sir[i]];
	}
  }
  else{
#pragma omp parallel for
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[i+b] += sr[sir[i]+b];
        }
	}
  }
}
