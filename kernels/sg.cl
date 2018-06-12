/* 
Generalized version of scatter and gather operations 
*/
#include "sgtype.h"
__kernel void sg(__global SGTYPE_CL* restrict target, 
                  __global long*   restrict ti,
                  __global SGTYPE_CL* restrict source,
                  __global long*   restrict si,
                  long ts, 
                  long ss, 
                  long n, 
                  long ws,
                  long R,
                  long B)
{
  //int s = ws - 1;
  int s = 0;
  __global SGTYPE_CL *tr, *sr;
  __global long   *tir, *sir;
  
  //not blocked version
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si    + s * (n);
	    for(long i = 0; i < n; i++){
	    	tr[tir[i]] SGOP sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }//TODO: Implement blocked version
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si + s * (n);
	    for(long i = 0; i < n; i++){
          for(long b = 0; b < B; b++){
	    	tr[tir[i]+b] SGOP sr[sir[i]+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}
