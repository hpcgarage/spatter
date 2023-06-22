// Author: Patrick Lavin
// Date:   14 January 2020

#include <stdint.h>

#ifndef MORTON_H
#define MORTON_H
uint32_t *z_order_1d(uint64_t dim, uint64_t block);
uint32_t *z_order_2d(uint64_t dim, uint64_t block);
uint32_t *z_order_3d(uint64_t dim, uint64_t block);

uint64_t next_pow2(uint64_t x);
uint32_t *get_cube(uint64_t, uint64_t);
#endif
