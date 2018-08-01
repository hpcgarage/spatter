#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void sg(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double*  sr = source + os;     
    __global long2*  sir = si     + oi / 2;
    __global long2*  tir = ti     + oi / 2;

    long2 sidx = sir[gid];
    long2 tidx = tir[gid];
    tr[tidx.s0] = sr[sidx.s0];
    tr[tidx.s1] = sr[sidx.s1];
}
