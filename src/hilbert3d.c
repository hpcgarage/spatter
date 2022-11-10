#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hilbert3d.h"
#include "morton.h"

#define ABS(i)    (i >= 0 ? i : -i)
#define SIGN(i)   (i >= 0 ? 1 : -1)
#define MIN(x, y) ((x < y) ? x : y)

struct curve_desc {
    long start[3];
    long base[7];
    point rot[8];
    long rev[8];
};

// Emit the description parsed by `parse_desc`
void print_desc(struct curve_desc desc)
{
    printf("# Base Pattern\n");
    printf("%2ld %2ld %2ld %2ld %2ld %2ld %2ld\n", desc.base[0], desc.base[1], desc.base[2], desc.base[3], desc.base[4], desc.base[5], desc.base[6]);
    printf("# Recursion Struction\n");
    for (int i = 0; i < 8; i++) {
        printf("%2ld %2ld %2ld\n", desc.rot[i].v[0], desc.rot[i].v[1], desc.rot[i].v[2]);
    }
    printf("# Flip after rotation?\n");
    printf("%ld %ld %ld %ld %ld %ld %ld %ld\n", desc.rev[0], desc.rev[1], desc.rev[2], desc.rev[3], desc.rev[4], desc.rev[5], desc.rev[6], desc.rev[7]);


}

// Parse the curve description file
void parse_desc(FILE *file, struct curve_desc* desc)
{
    int phase = 0;
    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {

        // Skip lines starting with #
        if (line[0] == '#') {
            continue;
        }

        switch(phase) {
            case 0:
                // Scan the base pattern
                sscanf(line, "%ld %ld %ld %ld %ld %ld %ld", &desc->base[0], &desc->base[1], &desc->base[2], &desc->base[3], &desc->base[4], &desc->base[5], &desc->base[6]);
                phase = 1;
                break;
            case 1:
                // Rotations
                sscanf(line, "%ld %ld %ld", &desc->rot[count].v[0], &desc->rot[count].v[1], &desc->rot[count].v[2]);
                if(++count == 8) {
                    phase = 2;
                }
                break;
            case 2:
                // Flip after rotation?
                sscanf(line, "%ld %ld %ld %ld %ld %ld %ld %ld", &desc->rev[0], &desc->rev[1], &desc->rev[2], &desc->rev[3], &desc->rev[4], &desc->rev[5], &desc->rev[6], &desc->rev[7]);
                return;
        }

    }
}

point *rotate_path(point *path, long len, point r, long flip)
{
    point *new_path = (point *)malloc(sizeof(point) * len);
    for (int i = 0; i < len; i++) {
        new_path[i].v[ABS(r.v[0]-1)] = path[i].v[0] * SIGN(r.v[0]);
        new_path[i].v[ABS(r.v[1]-1)] = path[i].v[1] * SIGN(r.v[1]);
        new_path[i].v[ABS(r.v[2]-1)] = path[i].v[2] * SIGN(r.v[2]);
    }

    if (flip) {
        for (long low = 0, high = len - 1; low < high; low++, high--) {
            point tmp = new_path[low];
            new_path[low] = new_path[high];
            new_path[high] = tmp;
        }
    }

    return new_path;
}

// Change the starting point of the path
void adjust_base(point *path, long len, point base)
{
   long d0 = base.p.x - path[0].p.x;
   long d1 = base.p.y - path[0].p.y;
   long d2 = base.p.z - path[0].p.z;

   for (long i = 0; i < len; i++) {
       path[i].p.x += d0;
       path[i].p.y += d1;
       path[i].p.z += d2;
   }
}

void positive_points(point *path, long path_len)
{
    long mx, my, mz;

    mx = path[0].p.x;
    my = path[0].p.y;
    mz = path[0].p.z;

    for (int i = 1; i < path_len; i++) {
        mx = MIN(mx, path[i].p.x);
        my = MIN(my, path[i].p.y);
        mz = MIN(mz, path[i].p.z);
    }

    for (int i = 0; i < path_len; i++) {
        path[i].p.x -= mx;
        path[i].p.y -= my;
        path[i].p.z -= mz;
    }
}

point *generate_path(struct curve_desc desc, long level, long *len)
{

    long dim_length = 1 << level;
    long npoints = dim_length * dim_length * dim_length;

    point *path = (point *)malloc(sizeof(point) * npoints);

    path[0].p.x = 0;
    path[0].p.y = 0;
    path[0].p.z = 0;

    long path_len = 1;
    for (int i = 0; i < level; i++) {

        point *child[8];

        //generate children
        for (int i = 0; i < 8; i++) {
            child[i] = rotate_path(path, path_len, desc.rot[i], desc.rev[i]);
        }

        //adjust children base
        for (int i = 0; i < 7; i++) {
            point last = child[i][path_len-1];
            switch(desc.base[i]) {
                case 1:
                    last.p.x++;
                    break;
                case -1:
                    last.p.x--;
                    break;
                case 2:
                    last.p.y++;
                    break;
                case -2:
                    last.p.y--;
                    break;
                case 3:
                    last.p.z++;
                    break;
                case -3:
                    last.p.z--;
                    break;
            }
            adjust_base(child[i+1], path_len, last);
        }

        //append children to list
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < path_len; j++) {
                path[i*path_len + j] = child[i][j];
            }
        }

        //free child lists
        for (int i = 0; i < 8; i++) {
            free(child[i]);
        }

        path_len *= 8;
    }

    assert(path_len == npoints);

    // Make all points positive
    positive_points(path, path_len);

    *len = npoints;
    return path;

}


void print_path(point *p, long len)
{
    for (int i = 0; i < len; i++) {
        printf("%2ld %2ld %2ld\n", p[i].p.x, p[i].p.y, p[i].p.z);
    }
}

point *hilbert3d_points(char *filename, long level, long *npoints)
{
    FILE *file;
    struct curve_desc desc;

    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Unable to open file\n");
        return NULL;
    }
    parse_desc(file, &desc);
    fclose(file);

    if (level < 0) {
        printf("Level must be non-negative\n");
        return NULL;
    }

    return generate_path(desc, level, npoints);

}

long *hilbert3d_indices(char *filename, long level, long *npoints)
{
    point *path = hilbert3d_points(filename, level, npoints);
    assert(path);

    long *indices = (long *)malloc(sizeof(long) * *npoints);
    long dim = 1 << level;

    for (int i = 0; i < *npoints; i++) {
         indices[i] = path[i].p.z * dim * dim +
                      path[i].p.y * dim +
                      path[i].p.x;
    }
    return indices;

}

uint32_t *h_order_3d(uint64_t dim_len, uint64_t block)
{
    long blocked_dim = dim_len / block;
    assert(blocked_dim * block == dim_len);

    long npoints;

    long level = next_pow2(blocked_dim);

    long *indices = hilbert3d_indices("beta.curve", level, &npoints);

    if (!indices) {
        return NULL;
    }

    uint32_t *cube = get_cube(dim_len, block);

    uint32_t *new_indices = (uint32_t*)malloc(sizeof(uint32_t) * dim_len * dim_len * dim_len);
    assert(new_indices);

    for (int i = 0; i < npoints; i++) {
        int x = indices[i] % blocked_dim;
        int y = indices[i] / blocked_dim % blocked_dim;
        int z = indices[i] / blocked_dim / blocked_dim;
        int base = x*block + y*dim_len*block + z*dim_len*dim_len*block;
        int off = i * (block*block*block);
        for (unsigned int j = 0; j < block*block*block; j++) {
            new_indices[off+j] = base+cube[j];
        }
    }

    free(indices);
    free(cube);

    return new_indices;
}
