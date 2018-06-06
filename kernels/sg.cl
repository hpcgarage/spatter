/* 
Generalized version of scatter and gather operations 
*/
__kernel void sg(__global double* restrict target, 
                  __global long*   restrict ti,
                  __global double* restrict source,
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
  __global double *tr, *sr;
  __global long   *tir, *sir;
  
  //not blocked version
  if(B == 1){
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
  }//TODO: Implement blocked version
  else{
    for(long r = 0; r < R; r++){
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si + s * (n);
	    for(long i = 0; i < n; i++){
          for(long b = 0; b < B; b++){
	    	tr[tir[i]+b] = sr[sir[i]+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}
