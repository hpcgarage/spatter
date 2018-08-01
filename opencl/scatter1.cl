#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void scatter(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot;
    __global double* sr = source + os;
    __global long *tir  = ti     + oi;
    tr[tir[gid]] = sr[gid];
}
