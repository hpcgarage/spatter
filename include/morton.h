// Author: Patrick Lavin
// Date:   14 January 2020

#include <stdint.h>

#ifndef MORTON_H
#define MORTON_H
uint32_t *z_order_1d(uint64_t dim);
uint32_t *z_order_2d(uint64_t dim);
uint32_t *z_order_3d(uint64_t dim);
#endif
