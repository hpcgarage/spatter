__kernel void sgp(__global double* restrict target, 
                  __global long*   restrict ti,
                  __global double* restrict source,
                  __global long*   restrict si,
                  __global long*            n,
                  __global long*            op)
{
	for(long i = 0; i < *n; i++){
		target[ti[i]] = source[si[i]];
	}
}
