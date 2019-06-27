#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void scatter1(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si)
{
    int gid = get_global_id(0);
    target[ti[gid]] = source[gid];
}

__kernel void scatter2(__global double* restrict target, 
                     __global double2*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si)
{
    int gid = get_global_id(0);

    double2 buf = source[gid];
    long2   idx = ti[gid];
    target[idx.s0] = buf.s0;
    target[idx.s1] = buf.s1;
}

__kernel void scatter4(__global double* restrict target, 
                     __global double4*  restrict source, 
                     __global long4*   restrict ti,
                     __global long4*   restrict si)
{
    int gid = get_global_id(0);

    double4 buf = source[gid];
    long4 idx = ti[gid];
    target[idx.s0] = buf.s0;
    target[idx.s1] = buf.s1;
    target[idx.s2] = buf.s2;
    target[idx.s3] = buf.s3;
}

__kernel void scatter8(__global double* restrict target, 
                     __global double8*  restrict source, 
                     __global long8*   restrict ti,
                     __global long8*   restrict si)
{
    int gid = get_global_id(0);

    double8 buf = source[gid];
    long8 idx = ti[gid];
    target[idx.s0] = buf.s0;
    target[idx.s1] = buf.s1;
    target[idx.s2] = buf.s2;
    target[idx.s3] = buf.s3;
    target[idx.s4] = buf.s4;
    target[idx.s5] = buf.s5;
    target[idx.s6] = buf.s6;
    target[idx.s7] = buf.s7;
}

__kernel void scatter16(__global double* restrict target, 
                     __global double16*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = get_global_id(0);

    double16 buf = source[gid];
    long16 idx = ti[gid];
    target[idx.s0] = buf.s0;
    target[idx.s1] = buf.s1;
    target[idx.s2] = buf.s2;
    target[idx.s3] = buf.s3;
    target[idx.s4] = buf.s4;
    target[idx.s5] = buf.s5;
    target[idx.s6] = buf.s6;
    target[idx.s7] = buf.s7;
    target[idx.s8] = buf.s8;
    target[idx.s9] = buf.s9;
    target[idx.sa] = buf.sa;
    target[idx.sb] = buf.sb;
    target[idx.sc] = buf.sc;
    target[idx.sd] = buf.sd;
    target[idx.se] = buf.se;
    target[idx.sf] = buf.sf;
}

__kernel void scatter32(__global double* restrict target, 
                     __global double16*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 2*get_global_id(0);

    double16 buf[2];
    long16 idx[2];

    for(int i = 0; i < 2; i++){
        buf[i] = source[gid+i];
        idx[i] = ti[gid+i];
    }

    for(int i = 0; i < 2; i++){
        target[idx[i].s0] = buf[i].s0;
        target[idx[i].s1] = buf[i].s1;
        target[idx[i].s2] = buf[i].s2;
        target[idx[i].s3] = buf[i].s3;
        target[idx[i].s4] = buf[i].s4;
        target[idx[i].s5] = buf[i].s5;
        target[idx[i].s6] = buf[i].s6;
        target[idx[i].s7] = buf[i].s7;
        target[idx[i].s8] = buf[i].s8;
        target[idx[i].s9] = buf[i].s9;
        target[idx[i].sa] = buf[i].sa;
        target[idx[i].sb] = buf[i].sb;
        target[idx[i].sc] = buf[i].sc;
        target[idx[i].sd] = buf[i].sd;
        target[idx[i].se] = buf[i].se;
        target[idx[i].sf] = buf[i].sf;
    }
}

__kernel void scatter64(__global double* restrict target, 
                     __global double16*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 4*get_global_id(0);

    double16 buf[4];
    long16 idx[4];

    for(int i = 0; i < 4; i++){
        buf[i] = source[gid+i];
        idx[i] = ti[gid+i];
    }
    for(int i = 0; i < 4; i++){
        target[idx[i].s0] = buf[i].s0;
        target[idx[i].s1] = buf[i].s1;
        target[idx[i].s2] = buf[i].s2;
        target[idx[i].s3] = buf[i].s3;
        target[idx[i].s4] = buf[i].s4;
        target[idx[i].s5] = buf[i].s5;
        target[idx[i].s6] = buf[i].s6;
        target[idx[i].s7] = buf[i].s7;
        target[idx[i].s8] = buf[i].s8;
        target[idx[i].s9] = buf[i].s9;
        target[idx[i].sa] = buf[i].sa;
        target[idx[i].sb] = buf[i].sb;
        target[idx[i].sc] = buf[i].sc;
        target[idx[i].sd] = buf[i].sd;
        target[idx[i].se] = buf[i].se;
        target[idx[i].sf] = buf[i].sf;
    }
}
__kernel void gather1(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si)
{
    int gid = get_global_id(0);
    target[gid] = source[si[gid]];
}

__kernel void gather2(__global double2* restrict target, 
                     __global double*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si)
{
    int gid = get_global_id(0);

    double2 buf = 0;
    long2 idx = si[gid];
    buf.s0 = source[idx.s0];
    buf.s1 = source[idx.s1];

    target[gid] = buf;
}

__kernel void gather4(__global double4* restrict target, 
                     __global double*  restrict source, 
                     __global long4*   restrict ti,
                     __global long4*   restrict si)
{
    int gid = get_global_id(0);

    double4 buf = 0;
    long4 idx = si[gid];
    buf.s0 = source[idx.s0];
    buf.s1 = source[idx.s1];
    buf.s2 = source[idx.s2];
    buf.s3 = source[idx.s3];

    target[gid] = buf;
}

__kernel void gather8(__global double8* restrict target, 
                     __global double*  restrict source, 
                     __global long8*   restrict ti,
                     __global long8*   restrict si)
{
    int gid = get_global_id(0);

    double8 buf = 0;
    long8 idx = si[gid];
    buf.s0 = source[idx.s0];
    buf.s1 = source[idx.s1];
    buf.s2 = source[idx.s2];
    buf.s3 = source[idx.s3];
    buf.s4 = source[idx.s4];
    buf.s5 = source[idx.s5];
    buf.s6 = source[idx.s6];
    buf.s7 = source[idx.s7];

    target[gid] = buf;
}

__kernel void gather16(__global double16* restrict target, 
                     __global double*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = get_global_id(0);

    double16 buf = 0;
    long16 idx = si[gid];
    buf.s0 = source[idx.s0];
    buf.s1 = source[idx.s1];
    buf.s2 = source[idx.s2];
    buf.s3 = source[idx.s3];
    buf.s4 = source[idx.s4];
    buf.s5 = source[idx.s5];
    buf.s6 = source[idx.s6];
    buf.s7 = source[idx.s7];
    buf.s8 = source[idx.s8];
    buf.s9 = source[idx.s9];
    buf.sa = source[idx.sa];
    buf.sb = source[idx.sb];
    buf.sc = source[idx.sc];
    buf.sd = source[idx.sd];
    buf.se = source[idx.se];
    buf.sf = source[idx.sf];

    target[gid] = buf;
}

__kernel void gather32(__global double16* restrict target, 
                     __global double*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 2*get_global_id(0);

    double16 buf[2]; 
    long16 idx[2]; 
    
    for(int i = 0; i < 2; i++){
        buf[i] = 0;
        idx[i] = si[gid+i];
    }
    
    for(int i = 0; i < 2; i++){
        buf[i].s0 = source[idx[i].s0];
        buf[i].s1 = source[idx[i].s1];
        buf[i].s2 = source[idx[i].s2];
        buf[i].s3 = source[idx[i].s3];
        buf[i].s4 = source[idx[i].s4];
        buf[i].s5 = source[idx[i].s5];
        buf[i].s6 = source[idx[i].s6];
        buf[i].s7 = source[idx[i].s7];
        buf[i].s8 = source[idx[i].s8];
        buf[i].s9 = source[idx[i].s9];
        buf[i].sa = source[idx[i].sa];
        buf[i].sb = source[idx[i].sb];
        buf[i].sc = source[idx[i].sc];
        buf[i].sd = source[idx[i].sd];
        buf[i].se = source[idx[i].se];
        buf[i].sf = source[idx[i].sf];
    }

    for(int i = 0; i < 2; i++){
        target[gid+i] = buf[i];
    }
}

__kernel void gather64(__global double16* restrict target, 
                     __global double*   restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 4*get_global_id(0);

    double16 buf[4]; 
    long16 idx[4]; 
    
    for(int i = 0; i < 4; i++){
        buf[i] = 0;
        idx[i] = si[gid+i];
    }
    
    for(int i = 0; i < 4; i++){
        buf[i].s0 = source[idx[i].s0];
        buf[i].s1 = source[idx[i].s1];
        buf[i].s2 = source[idx[i].s2];
        buf[i].s3 = source[idx[i].s3];
        buf[i].s4 = source[idx[i].s4];
        buf[i].s5 = source[idx[i].s5];
        buf[i].s6 = source[idx[i].s6];
        buf[i].s7 = source[idx[i].s7];
        buf[i].s8 = source[idx[i].s8];
        buf[i].s9 = source[idx[i].s9];
        buf[i].sa = source[idx[i].sa];
        buf[i].sb = source[idx[i].sb];
        buf[i].sc = source[idx[i].sc];
        buf[i].sd = source[idx[i].sd];
        buf[i].se = source[idx[i].se];
        buf[i].sf = source[idx[i].sf];
    }

    for(int i = 0; i < 4; i++){
        target[gid+i] = buf[i];
    }
}

__kernel void sg1(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si)
{
    int gid = get_global_id(0);
    target[ti[gid]] = source[si[gid]];
}

__kernel void sg2(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long2*   restrict ti,
                     __global long2*   restrict si)
{
    int gid = get_global_id(0);

    long2 sidx = si[gid];
    long2 tidx = ti[gid];
    target[tidx.s0] = source[sidx.s0];
    target[tidx.s1] = source[sidx.s1];
}

__kernel void sg4(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long4*   restrict ti,
                     __global long4*   restrict si)
{
    int gid = get_global_id(0);

    long4 sidx = si[gid];
    long4 tidx = ti[gid];
    target[tidx.s0] = source[sidx.s0];
    target[tidx.s1] = source[sidx.s1];
    target[tidx.s2] = source[sidx.s2];
    target[tidx.s3] = source[sidx.s3];
}

__kernel void sg8(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long8*   restrict ti,
                     __global long8*   restrict si)
{
    int gid = get_global_id(0);

    long8 sidx = si[gid];
    long8 tidx = ti[gid];
    target[tidx.s0] = source[sidx.s0];
    target[tidx.s1] = source[sidx.s1];
    target[tidx.s2] = source[sidx.s2];
    target[tidx.s3] = source[sidx.s3];
    target[tidx.s4] = source[sidx.s4];
    target[tidx.s5] = source[sidx.s5];
    target[tidx.s6] = source[sidx.s6];
    target[tidx.s7] = source[sidx.s7];
}

__kernel void sg16(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = get_global_id(0);

    long16 sidx = si[gid];
    long16 tidx = ti[gid];
    target[tidx.s0] = source[sidx.s0];
    target[tidx.s1] = source[sidx.s1];
    target[tidx.s2] = source[sidx.s2];
    target[tidx.s3] = source[sidx.s3];
    target[tidx.s4] = source[sidx.s4];
    target[tidx.s5] = source[sidx.s5];
    target[tidx.s6] = source[sidx.s6];
    target[tidx.s7] = source[sidx.s7];
    target[tidx.s8] = source[sidx.s8];
    target[tidx.s9] = source[sidx.s9];
    target[tidx.sa] = source[sidx.sa];
    target[tidx.sb] = source[sidx.sb];
    target[tidx.sc] = source[sidx.sc];
    target[tidx.sd] = source[sidx.sd];
    target[tidx.se] = source[sidx.se];
    target[tidx.sf] = source[sidx.sf];
}

__kernel void sg32(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 2*get_global_id(0);

    long16 sidx[2]; 
    long16 tidx[2];
    
    for(int i = 0; i < 2; i++){
        sidx[i] = si[gid+i];
        tidx[i] = ti[gid+i];
    }

    for(int i = 0; i < 2; i++){ 
        target[tidx[i].s0] = source[sidx[i].s0];
        target[tidx[i].s1] = source[sidx[i].s1];
        target[tidx[i].s2] = source[sidx[i].s2];
        target[tidx[i].s3] = source[sidx[i].s3];
        target[tidx[i].s4] = source[sidx[i].s4];
        target[tidx[i].s5] = source[sidx[i].s5];
        target[tidx[i].s6] = source[sidx[i].s6];
        target[tidx[i].s7] = source[sidx[i].s7];
        target[tidx[i].s8] = source[sidx[i].s8];
        target[tidx[i].s9] = source[sidx[i].s9];
        target[tidx[i].sa] = source[sidx[i].sa];
        target[tidx[i].sb] = source[sidx[i].sb];
        target[tidx[i].sc] = source[sidx[i].sc];
        target[tidx[i].sd] = source[sidx[i].sd];
        target[tidx[i].se] = source[sidx[i].se];
        target[tidx[i].sf] = source[sidx[i].sf];
    }
}

__kernel void sg64(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long16*   restrict ti,
                     __global long16*   restrict si)
{
    int gid = 4*get_global_id(0);

    long16 sidx[4]; 
    long16 tidx[4];
    
    for(int i = 0; i < 4; i++){
        sidx[i] = si[gid+i];
        tidx[i] = ti[gid+i];
    }

    for(int i = 0; i < 4; i++){ 
        target[tidx[i].s0] = source[sidx[i].s0];
        target[tidx[i].s1] = source[sidx[i].s1];
        target[tidx[i].s2] = source[sidx[i].s2];
        target[tidx[i].s3] = source[sidx[i].s3];
        target[tidx[i].s4] = source[sidx[i].s4];
        target[tidx[i].s5] = source[sidx[i].s5];
        target[tidx[i].s6] = source[sidx[i].s6];
        target[tidx[i].s7] = source[sidx[i].s7];
        target[tidx[i].s8] = source[sidx[i].s8];
        target[tidx[i].s9] = source[sidx[i].s9];
        target[tidx[i].sa] = source[sidx[i].sa];
        target[tidx[i].sb] = source[sidx[i].sb];
        target[tidx[i].sc] = source[sidx[i].sc];
        target[tidx[i].sd] = source[sidx[i].sd];
        target[tidx[i].se] = source[sidx[i].se];
        target[tidx[i].sf] = source[sidx[i].sf];
    }
}
