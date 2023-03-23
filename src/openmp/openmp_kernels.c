#include "pcg_basic.h"
#include "openmp_kernels.h"

#include <stdio.h>
#define SIMD 8

#if !defined( USE_OPENMP )
#define omp_get_thread_num() 0
#endif

void gather_smallbuf(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();

#ifdef __CRAYC__
    #pragma concurrent
#endif
#ifdef __INTEL_COMPILER
    #pragma ivdep
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + delta * i;
           sgData_t *tl = target[t] + pat_len*(i%target_len);
#ifdef __CRAYC__
    #pragma concurrent
#endif
#if defined __CRAYC__ || defined __INTEL_COMPILER
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
    }
}

void gather_smallbuf_morton(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        uint32_t *order) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();

#ifdef __CRAYC__
    #pragma concurrent
#endif
#ifdef __INTEL_COMPILER
    #pragma ivdep
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + delta * order[i];
           sgData_t *tl = target[t] + pat_len*(i%target_len);
#ifdef __CRAYC__
    #pragma concurrent
#endif
#if defined __CRAYC__ || defined __INTEL_COMPILER
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
    }
}
void gather_smallbuf_rdm(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();

#ifdef __CRAYC__
    #pragma concurrent
#endif
#ifdef __INTEL_COMPILER
    #pragma ivdep
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + rand()%((n-1)*delta);
           sgData_t *tl = target[t] + pat_len*(i%target_len);
#ifdef __CRAYC__
    #pragma concurrent
#endif
#if defined __CRAYC__ || defined __INTEL_COMPILER
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
    }
}

void scatter_smallbuf(
        sgData_t* restrict target,
        sgData_t** const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();

#ifdef __CRAYC__
    #pragma concurrent
#endif
#ifdef __INTEL_COMPILER
    #pragma ivdep
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i;
           sgData_t *sl = source[t] + pat_len*(i%source_len);
#ifdef __CRAYC__
    #pragma concurrent
#endif
#if defined __CRAYC__ || defined __INTEL_COMPILER
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[pat[j]] = sl[j];
           }
        }
    }
}

void gather_smallbuf_random(
        sgData_t** restrict target,
        sgData_t* const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len,
        long initstate) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, initstate, t);


#ifdef __CRAYC__
    #pragma concurrent
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
            //long r = ()%n;
           uint32_t r = pcg32_boundedrand_r(&rng, (uint32_t)n);
           sgData_t *sl = source + delta * r;
           sgData_t *tl = target[t] + pat_len*(i%target_len);
#ifdef __CRAYC__
    #pragma concurrent
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
    }
}

void scatter_smallbuf_random(
        sgData_t* restrict target,
        sgData_t** const restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t source_len,
        long initstate) {
    if (n > 1ll<<32) {printf("n too big for rng, exiting.\n"); exit(1);}
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();
        pcg32_random_t rng;
        pcg32_srandom_r(&rng, initstate, t);

#ifdef __CRAYC__
    #pragma concurrent
#endif
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           uint32_t r = pcg32_boundedrand_r(&rng, (uint32_t)n);
           sgData_t *tl = target + delta * r;
           sgData_t *sl = source[t] + pat_len*(i%source_len);
#ifdef __CRAYC__
    #pragma concurrent
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               tl[pat[j]] = sl[j];
           }
        }
    }
}


void gather_smallbuf_multidelta(
        sgData_t** restrict target,
        sgData_t* restrict source,
        sgIdx_t* const restrict pat,
        size_t pat_len,
        size_t *delta,
        size_t n,
        size_t target_len,
        size_t delta_len) {
#ifdef __GNUC__
    #pragma omp parallel
#else
    #pragma omp parallel shared(pat)
#endif
    {
        int t = omp_get_thread_num();
#ifdef __CRAYC__
    #pragma concurrent
#endif
        //taget_len is in multiples of pat_len
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + (i/delta_len)*delta[delta_len-1] + delta[i%delta_len] - delta[0];
           sgData_t *tl = target[t] + pat_len*(i%target_len);
           //sgData_t *sl = source;
           //sgData_t *tl = target[0];
#ifdef __CRAYC__
    #pragma concurrent
    #pragma vector always,unaligned
#endif
           for (size_t j = 0; j < pat_len; j++) {
               //printf("i: %zu, j: %zu\n", i, j);
               tl[j] = sl[pat[j]];
               //tl[j] = sl[pat[j]];
           }
        }
    }
}

