#include <omp.h>
#include "../cl-helper.h"
#include "../sgtype.h"
void sg_omp(SGTYPE* restrict target, 
            long*   restrict ti,
            SGTYPE* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  SGTYPE *tr, *sr;
  long   *tir, *sir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	    	tr[tir[i]] SGOP sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si    + s * (n);
	    for(long i = 0; i < n; i++){
          for(int b = 0; b < B; b++){
	    	tr[tir[i]+b] SGOP sr[sir[i]+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

void scatter_omp(SGTYPE* restrict target, 
            long*   restrict ti,
            SGTYPE* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  SGTYPE *tr, *sr;
  long   *tir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	        tr[tir[i]] SGOP sr[i];
        }
      s = ((s-1) % ws + ws) % ws;
    }
  }
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
	    for(long i = 0; i < n; i++){
          for(int b = 0; b < B; b++){
	        tr[tir[i]+b] SGOP sr[i+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

void gather_omp(SGTYPE* restrict target, 
            long*   restrict ti,
            SGTYPE* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  SGTYPE *tr, *sr;
  long   *sir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      sir = si    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	      tr[i] SGOP sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      sir = si    + s * (n);
	    for(long i = 0; i < n; i++){
          for(int b = 0; b < B; b++){
	    	tr[i+b] SGOP sr[sir[i]+b];
          } 
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

