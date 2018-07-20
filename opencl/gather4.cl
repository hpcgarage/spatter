#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void gather(__global double4* restrict target, 
                     __global double*  restrict source, 
                     __global long4*   restrict index,
                     __global long4*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double4* tr = target + ot / 4; 
    __global double*  sr = source + os;     
    __global long4*  sir = si     + oi / 4;

    double4 buf = 0;
    long4 idx = sir[gid];
    buf.s0 = sr[idx.s0];
    buf.s1 = sr[idx.s1];
    buf.s2 = sr[idx.s2];
    buf.s3 = sr[idx.s3];

    tr[gid] = buf;
}
