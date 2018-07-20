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

  size_t num_workers = get_global_size(0);
  size_t id          = get_global_id(0);

  size_t work_per_item = n / num_workers;
  size_t my_work = work_per_item;
  if (id == num_workers-1) 
    my_work = my_work + (n % num_workers);

  if (id != 0)
    size_t start = work_per_item * id-1;
  else
    start = 0;

  size_t end = start + my_work;

  printf("start %zu end %zu\n", start, end);

  if (B == 1) {
    for (long r = 0; r < R; r++) {
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si    + s * (n);
	    for(long i = start; i < end; i++){
	    	tr[tir[i]] = sr[sir[i]];
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
  else{
    for (long r = 0; r < R; r++) {
      tr = target + s * (ts);
      sr = source + s * (ss);
      tir = ti    + s * (n);
      sir = si + s * (n);
	    for (long i = start; i < end; i++) {
          for (long b = 0; b < B; b++) {
	    	tr[tir[i]+b] = sr[sir[i]+b];
          }
	    }
      s = ((s-1) % ws + ws) % ws;
    }
  }
}
