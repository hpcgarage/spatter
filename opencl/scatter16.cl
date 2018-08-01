#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void scatter(__global double16* restrict target, 
                     __global double*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double16*  sr = source + os / 16;     
    __global long16*  tir = ti     + oi / 16;

    double16 buf = sr[gid];
    long16 idx = tir[gid];
    tr[idx.s0] = buf.s0;
    tr[idx.s1] = buf.s1;
    tr[idx.s2] = buf.s2;
    tr[idx.s3] = buf.s3;
    tr[idx.s4] = buf.s4;
    tr[idx.s5] = buf.s5;
    tr[idx.s6] = buf.s6;
    tr[idx.s7] = buf.s7;
    tr[idx.s8] = buf.s8;
    tr[idx.s9] = buf.s9;
    tr[idx.sa] = buf.sa;
    tr[idx.sb] = buf.sb;
    tr[idx.sc] = buf.sc;
    tr[idx.sd] = buf.sd;
    tr[idx.se] = buf.se;
    tr[idx.sf] = buf.sf;
}
