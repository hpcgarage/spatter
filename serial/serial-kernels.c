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

  if(B == 1){
#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[sir[i]];
	}
  }
  else{
	for(long i = 0; i < n; i++){
	#pragma novector
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[sir[i]+b];
        }
	}
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

  if(B == 1){
#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] = sr[i];
	}
  }
  else{
	for(long i = 0; i < n; i++){
	#pragma novector
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] = sr[i+b];
        }
	}
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


  if(B == 1){
#pragma novector
	for(long i = 0; i < n; i++){
	    tr[i] = sr[sir[i]];
	}
  }
  else{
	for(long i = 0; i < n; i++){
	#pragma novector
        for(int b = 0; b < B; b++){
	        tr[i+b] = sr[sir[i]+b];
        }
	}
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


  if(B == 1){
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[sir[i]];
	}
  }
  else{
#pragma novector
	for(long i = 0; i < n; i++){
	for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[sir[i]+b];
        }
	}
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

  if(B == 1){
#pragma novector
	for(long i = 0; i < n; i++){
	    tr[tir[i]] += sr[i];
	}
  }
  else{
#pragma novector
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[tir[i]+b] += sr[i+b];
        }
	}
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


  if(B == 1){
#pragma novector
	for(long i = 0; i < n; i++){
	    tr[i] += sr[sir[i]];
	}
  }
  else{
#pragma novector
	for(long i = 0; i < n; i++){
        for(int b = 0; b < B; b++){
	        tr[i+b] += sr[sir[i]+b];
        }
	}
  }
}
