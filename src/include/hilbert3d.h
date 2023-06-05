#ifndef HILBERT3D_H
#define HILBERT3D_H

#include <stdint.h>

typedef union point {
    long v[3];
    struct _p {
        long x;
        long y;
        long z;
    } p;
} point;

point *hilbert3d_points(char *filename, long level, long *npoints);

void print_path(point *path, long len);

long *hilbert3d_indices(char *filename, long level, long *npoints);

uint32_t *h_order_3d(uint64_t dim, uint64_t block);
#endif
