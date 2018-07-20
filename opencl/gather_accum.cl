#include "sgtype.h"
__kernel void gather_accum(__global SGTYPE_CL* restrict target, 
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
  __global long   *sir;
  if(B == 1){
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      sir = si    + s * (n);
	    for(long i = 0; i < n; i++){
	    	tr[i] += sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      sir = si + s * (n);
	    for(long i = 0; i < n; i++){
          for(long b = 0; b < B; b++){
	    	tr[i+b] += sr[sir[i]+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}
