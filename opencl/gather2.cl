#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
__kernel void gather(__global double2* restrict target, 
                     __global double*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double2* tr = target + ot / 2; 
    __global double*  sr = source + os;     
    __global long2*  sir = si     + oi / 2;

    double2 buf = 0;
    long2 idx = sir[gid];
    buf.s0 = sr[idx.s0];
    buf.s1 = sr[idx.s1];

    tr[gid] = buf;
}
