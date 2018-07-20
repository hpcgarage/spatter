#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void gather(__global double8* restrict target, 
                     __global double*  restrict source, 
                     __global long8*   restrict index,
                     __global long8*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double8* tr = target + ot / 8; 
    __global double*  sr = source + os;     
    __global long8*  sir = si     + oi / 8;

    double8 buf = 0;
    long8 idx = sir[gid];
    buf.s0 = sr[idx.s0];
    buf.s1 = sr[idx.s1];
    buf.s2 = sr[idx.s2];
    buf.s3 = sr[idx.s3];
    buf.s4 = sr[idx.s4];
    buf.s5 = sr[idx.s5];
    buf.s6 = sr[idx.s6];
    buf.s7 = sr[idx.s7];

    tr[gid] = buf;
}
