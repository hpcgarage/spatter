#define _GNU_SOURCE
#include <string.h> //memset
#include <stdlib.h> //exit
#include <stdio.h>
#include "sp_alloc.h"
#include "parse-args.h" //error
#include <stdio.h>


long long total_mem_used = 0;

long long get_mem_used() {
    return total_mem_used;
}
void check_size(size_t size) {
    total_mem_used += size;
    //printf("size: %zu\n", size);
    if (total_mem_used > SP_MAX_ALLOC) {
        error("Too much memory used.", ERROR);
    }
}

void check_safe_mult(size_t a, size_t b) {
    unsigned int hi_bit_a = 0;
    unsigned int hi_bit_b = 0;

    while (a >>= 1) hi_bit_a++;
    while (b >>= 1) hi_bit_b++;

    if (hi_bit_a + hi_bit_b > sizeof(size_t) * 8) {
        error("Error: Multiplication would overflow.", ERROR);
    }

}

void *sp_malloc (size_t size, size_t count, size_t align) {
    check_safe_mult(size, count);
    check_size(size*count);
#ifdef USE_POSIX_MEMALIGN
    void *ptr = NULL;
    int ret = posix_memalign (&ptr,align,size*count);
    if (ret!=0) ptr = NULL;
#else
    void *ptr = aligned_alloc (align, size*count);
#endif
    if (!ptr) {
        printf("Attempted to allocate %zu bytes (%zu * %zu)\n", size*count, size , count);
        error("Error: failed to allocate memory", ERROR);
    }
    return ptr;
}

void *sp_calloc (size_t size, size_t count, size_t align) {
    void *ptr = sp_malloc(size, count, align);
    memset(ptr, 0, size*count);
    return ptr;
}
