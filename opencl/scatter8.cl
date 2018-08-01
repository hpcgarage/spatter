#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void scatter(__global double8* restrict target, 
                     __global double*  restrict source, 
                     __global long8*   restrict ti,
                     __global long8*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double8*  sr = source + os / 8;     
    __global long8*  tir = ti     + oi / 8;

    double8 buf = sr[gid];
    long8 idx = tir[gid];
    tr[idx.s0] = buf.s0;
    tr[idx.s1] = buf.s1;
    tr[idx.s2] = buf.s2;
    tr[idx.s3] = buf.s3;
    tr[idx.s4] = buf.s4;
    tr[idx.s5] = buf.s5;
    tr[idx.s6] = buf.s6;
    tr[idx.s7] = buf.s7;
}
