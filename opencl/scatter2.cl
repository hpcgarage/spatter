#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void scatter(__global double2* restrict target, 
                     __global double*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double2*  sr = source + os /2;     
    __global long2*  tir = ti     + oi / 2;

    double2 buf = sr[gid];
    long2   idx = tir[gid];
    tr[idx.s0] = buf.s0;
    tr[idx.s1] = buf.s1;
}
