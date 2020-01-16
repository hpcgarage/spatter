// Author: Patrick Lavin
// Date:   14 January 2020
//
// This code returns a 2d or 3d morton ordering


#define MORTON_VERBOSE 1
#define MORTON_DEBUG 0

#ifdef MORTON_VERBOSE
#include <stdio.h>
#endif
#include <stdlib.h>
#include <stdint.h>
#include "morton.h"
#include "sp_alloc.h"

uint64_t next_pow2(uint64_t x)
{
    uint64_t n = 0, xx = x;
    while (xx >>= 1) n++;
    if (1 << n != x) n++;
    return n;
}

// Taken from https://stackoverflow.com/a/30562230
uint64_t even_bits(uint64_t x)
{
    x = x & 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return (uint64_t)x;
}

void unpack_2d(uint64_t d, uint64_t *x, uint64_t *y)
{
    *x = even_bits(d);
    *y = even_bits(d >> 1);
}

uint32_t *z_order_2d(uint64_t dim)
{
    uint64_t i, x, y, idx = 0, extra = 0;
    uint32_t* list = NULL;

    if (dim == 0) {
#if MORTON_VERBOSE
        printf("Error: dim must be positive\n");
#endif
        return NULL;
    }

    if (next_pow2(dim) > 32) {
#if MORTON_VERBOSE
        printf("Error: The dimension is too big to be mixed in 2d\n");
#endif
        return NULL;
    }

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim * dim, ALIGN_CACHE);

    if (!list) {
#if MORTON_VERBOSE
        printf("Failed to allocate space for the ordering\n");
#endif
        return NULL;
    }

    for (i = 0; i < dim * dim + extra; i++) {
        unpack_2d(i, &y, &x);
        if (x >= dim || y >= dim) {
            extra++;
            continue;
        }
        list[idx++] = (uint32_t)(x*dim + y);

        #if MORTON_DEBUG
            printf("(%lu,%lu) -> %lu\n", x, y, ((uint64_t)x)*dim + y);
        #endif

    }

    return list;
}


// Taken from https://stackoverflow.com/a/28358035
uint64_t third_bits(uint64_t x) {
    x = x & 0x9249249249249249;
    x = (x | (x >> 2))  & 0x30C30C30C30C30C3;
    x = (x | (x >> 4))  & 0xF00F00F00F00F00F;
    x = (x | (x >> 8))  & 0x00FF0000FF0000FF;
    x = (x | (x >> 16)) & 0xFFFF00000000FFFF;
    x = (x | (x >> 32)) & 0x00000000FFFFFFFF;
    return x;
}

void unpack_3d(uint64_t d, uint64_t *x, uint64_t *y, uint64_t *z)
{
    *x = third_bits(d);
    *y = third_bits(d>>1);
    *z = third_bits(d>>2);
}

uint32_t *z_order_3d(uint64_t dim)
{
    uint64_t i, x, y, z, idx = 0, extra = 0;
    uint32_t *list = NULL;

    if (dim == 0) {
#if MORTON_VERBOSE
        printf("Error: dim must be positive\n");
#endif
        return NULL;
    }

    if (next_pow2(dim) > 21) {
#if MORTON_VERBOSE
        printf("Error: The dimension is too big to be mixed in 3d\n");
#endif
        return NULL;
    }

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim * dim * dim, ALIGN_PAGE);

    if (!list) {
#if MORTON_VERBOSE
        printf("Failed to allocate space for the ordering\n");
#endif
        return NULL;
    }

    for (i = 0; i < dim * dim * dim + extra; i++) {
        unpack_3d(i, &z, &y, &x);
        if (x >= dim || y >= dim || z >= dim) {
            extra++;
            continue;
        }
        list[idx++] = (uint32_t)(x*dim*dim + y*dim + z);

        #if MORTON_DEBUG
            printf("(%lu,%lu,%lu) -> %lu\n", x, y, z, x*dim*dim + y*dim + z);
        #endif
    }

    return list;
}

uint32_t *z_order_1d(uint64_t dim)
{
    uint64_t i;
    uint32_t *list = NULL;

    if (dim == 0) {
#if MORTON_VERBOSE
        printf("Error: dim must be positive\n");
#endif
        return NULL;
    }

    if (next_pow2(dim) > 32) {
#if MORTON_VERBOSE
        printf("Error: The dimension is too big - unsupported as return type too small\n");
#endif
        return NULL;
    }

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim, ALIGN_PAGE);

    if (!list) {
#if MORTON_VERBOSE
        printf("Failed to allocate space for the ordering\n");
#endif
        return NULL;
    }

    for (i = 0; i < dim; i++) {
        list[i] = i;
    }

    return list;
}
