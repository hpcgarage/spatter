#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void sg(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long8*   restrict ti,
                     __global long8*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double*  sr = source + os;     
    __global long8*  sir = si     + oi / 8;
    __global long8*  tir = ti     + oi / 8;

    long8 sidx = sir[gid];
    long8 tidx = tir[gid];
    tr[tidx.s0] = sr[sidx.s0];
    tr[tidx.s1] = sr[sidx.s1];
    tr[tidx.s2] = sr[sidx.s2];
    tr[tidx.s3] = sr[sidx.s3];
    tr[tidx.s4] = sr[sidx.s4];
    tr[tidx.s5] = sr[sidx.s5];
    tr[tidx.s6] = sr[sidx.s6];
    tr[tidx.s7] = sr[sidx.s7];
}
