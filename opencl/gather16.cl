#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void gather(__global double16* restrict target, 
                     __global double*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double16* tr = target + ot / 16; 
    __global double*  sr = source + os;     
    __global long16*  sir = si     + oi / 16;

    double16 buf = 0;
    long16 idx = sir[gid];
    buf.s0 = sr[idx.s0];
    buf.s1 = sr[idx.s1];
    buf.s2 = sr[idx.s2];
    buf.s3 = sr[idx.s3];
    buf.s4 = sr[idx.s4];
    buf.s5 = sr[idx.s5];
    buf.s6 = sr[idx.s6];
    buf.s7 = sr[idx.s7];
    buf.s8 = sr[idx.s8];
    buf.s9 = sr[idx.s9];
    buf.sa = sr[idx.sa];
    buf.sb = sr[idx.sb];
    buf.sc = sr[idx.sc];
    buf.sd = sr[idx.sd];
    buf.se = sr[idx.se];
    buf.sf = sr[idx.sf];

    tr[gid] = buf;
}
