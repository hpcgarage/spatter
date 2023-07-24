#include "serial-kernels.h"
#include <stdlib.h>

void multigather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        ssize_t* const restrict outer_pat,
        ssize_t* const restrict inner_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len) {

    for (size_t i = 0; i < n; i++) {
        sgData_t *sl = source + delta * i;
        //Pick which 8 elements are written to in a way that 
        //is hard to optimize out with a compiler
        sgData_t *tl = target[0] + pat_len*(i%target_len);

        #pragma novector    
        for (size_t j = 0; j < pat_len; j++) {
            tl[j] = sl[outer_pat[inner_pat[j]]];
        }
    }
}

void multiscatter_smallbuf_serial(
      sgData_t* restrict target,
      sgData_t** restrict source,
      ssize_t* const restrict outer_pat,
      ssize_t* const restrict inner_pat,
      size_t pat_len,
      size_t delta,
      size_t n,
      size_t source_len) {

    for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i;
           sgData_t *sl = source[0] + pat_len*(i%source_len);

       #pragma novector 
       for (size_t j = 0; j < pat_len; j++) {
               tl[outer_pat[inner_pat[j]]] = sl[j];
           }
        }
}

//Small index buffer version of gather
void gather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len) {

    for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + delta * i;
           //Pick which 8 elements are written to in a way that 
       //is hard to optimize out with a compiler
       sgData_t *tl = target[0] + pat_len*(i%target_len);

#ifndef __clang__
       #pragma novector
#endif
       for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
}


void scatter_smallbuf_serial(
        sgData_t* restrict target,
        sgData_t** const restrict source,
        ssize_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len) {

    for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i;
           sgData_t *sl = source[0] + pat_len*(i%source_len);

#ifndef __clang__
       #pragma novector
#endif
       for (size_t j = 0; j < pat_len; j++) {
               tl[pat[j]] = sl[j];
           }
        }
}


void sg_smallbuf_serial(
        sgData_t* restrict gather,
        sgData_t* restrict scatter,
        ssize_t* const restrict gather_pat,
        ssize_t* const restrict scatter_pat,
        size_t pat_len,
        size_t delta_gather,
        size_t delta_scatter,
        size_t n,
        size_t wrap) {

    for (size_t i = 0; i < n; i++) {
        sgData_t *tl = scatter + delta_scatter * i;
        sgData_t *sl = gather + delta_gather * i;

        #pragma novector
        for (size_t j = 0; j < pat_len; j++) {
            tl[scatter_pat[j]] = sl[gather_pat[j]];
        }
    }
}
