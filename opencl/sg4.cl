#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void sg(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long4*   restrict ti,
                     __global long4*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double*  sr = source + os;     
    __global long4*  sir = si     + oi / 4;
    __global long4*  tir = ti     + oi / 4;

    long4 sidx = sir[gid];
    long4 tidx = tir[gid];
    tr[tidx.s0] = sr[sidx.s0];
    tr[tidx.s1] = sr[sidx.s1];
    tr[tidx.s2] = sr[sidx.s2];
    tr[tidx.s3] = sr[sidx.s3];
}
