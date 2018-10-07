#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void scatter1(__global double* restrict target, 
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

__kernel void scatter2(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 2*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[2];
    long   idx[2];

    for(int i = 0; i < 2; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 2; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 2; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void scatter4(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 4*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[4];
    long   idx[4];

    for(int i = 0; i < 4; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 4; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 4; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void scatter8(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 8*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[8];
    long   idx[8];

    for(int i = 0; i < 8; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 8; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 8; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void scatter16(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 16*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[16];
    long   idx[16];

    for(int i = 0; i < 16; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 16; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 16; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void scatter32(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 32*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[32];
    long   idx[32];

    for(int i = 0; i < 32; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 32; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 32; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void scatter64(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 64*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  tir = ti     + oi;

    double buf[64];
    long   idx[64];

    for(int i = 0; i < 64; i++){
        buf[i] = sr[gid+i];
    }

    for(int i = 0; i < 64; i++){
        idx[i] = tir[gid+i];
    }

    for(int i = 0; i < 64; i++){
        tr[idx[i]] = buf[i];
    }
}

__kernel void gather1(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot;
    __global double* sr = source + os;
    __global long *sir  = si     + oi;
    tr[gid] = sr[sir[gid]];
}

__kernel void gather2(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 2*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[2];

    for(int i = 0; i < 2; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 2; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void gather4(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 4*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[4];

    for(int i = 0; i < 4; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 4; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void gather8(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 8*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[8];

    for(int i = 0; i < 8; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 8; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void gather16(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 16*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[16];

    for(int i = 0; i < 16; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 16; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void gather32(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 32*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[32];

    for(int i = 0; i < 32; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 32; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void gather64(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 64*get_global_id(0);
    __global double*  tr = target + ot; 
    __global double*  sr = source + os;     
    __global long*   sir = si     + oi;

    double buf[64];

    for(int i = 0; i < 64; i++){
        buf[i] = sr[sir[gid+i]];
    }

    for(int i = 0; i < 64; i++){
        tr[gid+i] = buf[i];
    }
}

__kernel void sg1(__global double* restrict target, 
                     __global double* restrict source, 
                     __global long* restrict ti,
                     __global long* restrict si,
                     long ot, long os, long oi)
{
    int gid = get_global_id(0);
    __global double* tr = target + ot;
    __global double* sr = source + os;
    __global long *sir  = si     + oi;
    __global long *tir  = ti     + oi;
    tr[tir[gid]] = sr[sir[gid]];
}

__kernel void sg2(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 2*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[2];
    long tidx[2];

    for(int i = 0; i < 2; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 2; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 2; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }

}

__kernel void sg4(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 4*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[4];
    long tidx[4];

    for(int i = 0; i < 4; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 4; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 4; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }

}

__kernel void sg8(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 8*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[8];
    long tidx[8];

    for(int i = 0; i < 8; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 8; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 8; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }

}

__kernel void sg16(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 16*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[16];
    long tidx[16];

    for(int i = 0; i < 16; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 16; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 16; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }
}

__kernel void sg32(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 32*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[32];
    long tidx[32];

    for(int i = 0; i < 32; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 32; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 32; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }
}

__kernel void sg64(__global double* restrict target, 
                     __global double*  restrict source, 
                     __global long*   restrict ti,
                     __global long*   restrict si,
                     long ot, long os, long oi)
{
    int gid = 64*get_global_id(0);
    __global double* tr = target + ot; 
    __global double* sr = source + os;     
    __global long*  sir = si     + oi;
    __global long*  tir = ti     + oi;

    long sidx[64];
    long tidx[64];

    for(int i = 0; i < 64; i++){
        sidx[i] = sir[gid+i];
    }
    
    for(int i = 0; i < 64; i++){
        tidx[i] = tir[gid+i];
    }

    for(int i = 0; i < 64; i++){
        tr[tidx[i]] = sr[sidx[i]];
    }
}
