#include <omp.h>

void sg_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  double *tr, *sr;
  long   *tir, *sir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	    	tr[tir[i]] = sr[sir[i]];
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
	    	tr[tir[i]] = sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

void scatter_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  double *tr, *sr;
  long   *tir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	    	tr[tir[i]] = sr[i];
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
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

void gather_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B)
{
  int s = 0;
  double *tr, *sr;
  long   *sir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      sir = si    + s * (n);
#pragma omp parallel for
	    for(long i = 0; i < n; i++){
	    	tr[i] = sr[sir[i]];
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
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}

