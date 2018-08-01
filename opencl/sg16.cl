#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void sg(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot; 
    __global double*  sr = source + os;     
    __global long16*  sir = si     + oi / 16;
    __global long16*  tir = ti     + oi / 16;

    long16 sidx = sir[gid];
    long16 tidx = tir[gid];
    tr[tidx.s0] = sr[sidx.s0];
    tr[tidx.s1] = sr[sidx.s1];
    tr[tidx.s2] = sr[sidx.s2];
    tr[tidx.s3] = sr[sidx.s3];
    tr[tidx.s4] = sr[sidx.s4];
    tr[tidx.s5] = sr[sidx.s5];
    tr[tidx.s6] = sr[sidx.s6];
    tr[tidx.s7] = sr[sidx.s7];
    tr[tidx.s8] = sr[sidx.s8];
    tr[tidx.s9] = sr[sidx.s9];
    tr[tidx.sa] = sr[sidx.sa];
    tr[tidx.sb] = sr[sidx.sb];
    tr[tidx.sc] = sr[sidx.sc];
    tr[tidx.sd] = sr[sidx.sd];
    tr[tidx.se] = sr[sidx.se];
    tr[tidx.sf] = sr[sidx.sf];
}
