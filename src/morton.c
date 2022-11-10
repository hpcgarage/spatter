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
#include "unused.h"

// Return the smallest power of 2 greater than or equal to `x`
uint64_t next_pow2(uint64_t x)
{
    uint64_t n = 0, xx = x;
    while (xx >>= 1) n++;
    if (1 << n != x) n++;
    return n;
}

// Return the even bits of `x`, packed into the low 32 bits
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

// Translate the morton coordinate d into a 2d x and y coordinate
void unpack_2d(uint64_t d, uint64_t *x, uint64_t *y)
{
    *x = even_bits(d);
    *y = even_bits(d >> 1);
}

uint32_t *get_square(uint64_t dim, uint64_t block)
{
    uint32_t *square = (uint32_t*)malloc(sizeof(uint32_t) * block * block);
    for (uint64_t i = 0; i < block; i++) {
        for (uint64_t j = 0; j < block; j++) {
            square[i*block + j] = i*dim + j;
        }
    }
    return square;
}

uint32_t *_z_block_2d(uint32_t* old_list, uint64_t dim, uint64_t block) {

    uint64_t i;
    uint32_t* list = NULL;
    uint32_t* square = NULL;

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim * dim, ALIGN_PAGE);

    square = get_square(dim, block);

    for (i = 0; i < (dim/block)*(dim/block); i++) {
        int x = old_list[i] % (dim/block);
        int y = old_list[i] / (dim/block);
        int base = x * block + y * dim * block;
        //int base = old_list[i] * (block * block);
        int off  = i * (block * block);
        for (uint64_t j = 0; j < block*block; j++) {
            list[off+j] = base + square[j];
        }
    }

    free(square);
    free(old_list);
    return list;
}

uint32_t *_z_order_2d(uint64_t dim)
{

    uint64_t i, x, y, idx = 0, extra = 0;
    uint32_t* list = NULL;

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim * dim, ALIGN_PAGE);

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

uint32_t *z_order_2d(uint64_t dim, uint64_t block)
{
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

    if (block <= 0) {
#if MORTON_VERBOSE
        printf("Error: The block size must be positive\n");
#endif
        return NULL;
    }

    if ((dim/block)*block != dim) {
#if MORTON_VERBOSE
        printf("Error: The block size must divide the dimension length\n");
#endif
        return NULL;
    }

    list = _z_order_2d(dim/block);
    if(block>1) list = _z_block_2d(list, dim, block);

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

uint32_t *get_cube(uint64_t dim, uint64_t block)
{
    uint32_t *cube = (uint32_t*)malloc(sizeof(uint32_t) * block * block * block);

    if (!cube) {
#if MORTON_VERBOSE
        printf("Failed to allocate space for the cube pattern\n");
#endif
        return NULL;
    }

    for (uint64_t i = 0; i < block; i++) {
        for (uint64_t j = 0; j < block; j++) {
            for (uint64_t k = 0; k < block; k++) {
                cube[i*block*block + j*block + k] = i*dim*dim + j*dim + k;
            }
        }
    }
    return cube;
}

uint32_t *_z_block_3d(uint32_t* old_list, uint64_t dim, uint64_t block) {

    uint64_t i;
    uint32_t* list = NULL;
    uint32_t* cube = NULL;

    list = (uint32_t*)sp_malloc(sizeof(uint32_t), dim * dim * dim, ALIGN_PAGE);

    if (!list) {
#if MORTON_VERBOSE
        printf("Failed to allocate space for the ordering\n");
#endif
        return NULL;
    }

    cube = get_cube(dim, block);

    for (i = 0; i < (dim/block)*(dim/block)*(dim/block); i++) {
        int x = old_list[i] % (dim/block);
        int y = old_list[i] / (dim/block) % (dim/block);
        int z = old_list[i] / (dim/block) / (dim/block);
        int base = x * block + y * dim * block + z * dim * dim * block;
        //int base = old_list[i] * (block * block);
        int off  = i * (block * block * block);
        for (uint64_t j = 0; j < block*block*block; j++) {
            list[off+j] = base + cube[j];
        }
    }

    free(cube);
    free(old_list);
    return list;
}

uint32_t *_z_order_3d(uint64_t dim)
{
    uint64_t i, x, y, z, idx = 0, extra = 0;
    uint32_t *list = NULL;

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

uint32_t *z_order_3d(uint64_t dim, uint64_t block)
{
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

    if (block <= 0) {
#if MORTON_VERBOSE
        printf("Error: The block size must be positive\n");
#endif
        return NULL;
    }

    if ((dim/block)*block != dim) {
#if MORTON_VERBOSE
        printf("Error: The block size must divide the dimension length\n");
#endif
        return NULL;
    }

    list = _z_order_3d(dim/block);
    if(block>1) list = _z_block_3d(list, dim, block);

    return list;
}

// TODO: Removed unused block paramemter. I believe it is only
// to match the signature of the other z_order_* methods. Unless
// we are using a function pointer somewhere we shouldn't need it.
uint32_t *z_order_1d(uint64_t dim, uint64_t UNUSED(block))
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
