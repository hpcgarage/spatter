#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void scatter(__global double4* restrict target, 
                     __global double*  restrict source, 
                     __global long4*   restrict ti,
                     __global long4*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double4*  sr = source + os / 4;     
    __global long4*  tir = ti     + oi / 4;

    double4 buf = sr[gid];
    long4 idx = tir[gid];
    tr[idx.s0] = buf.s0;
    tr[idx.s1] = buf.s1;
    tr[idx.s2] = buf.s2;
    tr[idx.s3] = buf.s3;
}
