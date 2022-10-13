#include "serial-kernels.h"
//Small index buffer version of gather
void gather_smallbuf_serial(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        sgIdx_t* const restrict pat,
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
               tl[j] = sl[pat[j]];
           }
        }
}


void scatter_smallbuf_serial(
        sgData_t* restrict target,
        sgData_t** const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len) {

    for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i;
           sgData_t *sl = source[0] + pat_len*(i%source_len);

       #pragma novector 
       for (size_t j = 0; j < pat_len; j++) {
               tl[pat[j]] = sl[j];
           }
        }
}


void sg_smallbuf_serial(
        sgData_t* restrict gather,
        sgData_t* restrict scatter,
        sgIdx_t* const restrict gather_pat,
        sgIdx_t* const restrict scatter_pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap) {

    for (size_t i = 0; i < n; i++) {
        sgData_t *tl = scatter + delta * i;
        sgData_t *sl = gather + delta * i;

        #pragma novector
        for (size_t j = 0; j < pat_len; j++) {
            tl[scatter_pat[j]] = sl[gather_pat[j]];
        }
    }
}
