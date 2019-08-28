#include "pcg_basic.h"
#include "openmp_kernels.h"

#include <stdio.h>
#define SIMD 8


void sg_omp(
            sgData_t* restrict target,   
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t             n)
{
#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
    for(long i = 0; i < n; i++){
        target[ti[i]] = source[si[i]];
	}
}

void scatter_omp(
            sgData_t* restrict target, 
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n)
{

#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[i];
	}

}

void gather_omp(
            sgData_t* restrict target, 
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n)
{

//Users may want to set a specific safelen value like 32
#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
	for(long i = 0; i < n; i++){
	    target[i] = source[si[i]];
	}
}

int get_ind(void) {
	return 1;
}

void gather(
		sgData_t* restrict target, 
		sgData_t* restrict source, 
		sgIdx_t*  const restrict pat, 
		size_t    pat_len,
		size_t    delta, 
		size_t    n)
{

#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
    for (size_t i = 0; i < n; i++) {
#pragma loop_info est_trips(8)
#pragma loop_info prefetch
        for (size_t j = 0; j < pat_len; j++) {
            target[i*pat_len+j] = source[pat[j]];
        }
        source += delta;
    }
}

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
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *sl = source + delta * i; 
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
#pragma omp for
        for (size_t i = 0; i < n; i++) {
           sgData_t *tl = target + delta * i; 
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
		
void scatter(
		sgData_t* restrict target, 
		sgData_t* restrict source, 
		sgIdx_t*  const restrict pat, 
		size_t    pat_len,
		size_t    delta, 
		size_t    n)
{

#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
    for (size_t i = 0; i < n; i++) {
#pragma loop_info est_trips(8)
#pragma loop_info prefetch
        for (size_t j = 0; j < pat_len; j++) {
            target[pat[j]] = source[i*pat_len+j];
        }
        source += delta;
    }
}
void gather_stride_os(
		sgData_t* restrict target, 
		sgData_t* restrict source, 
		sgIdx_t*  restrict pat, 
		size_t    pat_len,
		size_t    delta, 
		size_t    n, 
        size_t    target_wrap)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < pat_len; j++) {
            target[(i%target_wrap)*pat_len+j] = source[pat[j]];
        }
        source += delta;
    }
}
		
		
void gather_stride8(
		sgData_t* restrict target, 
		sgData_t* restrict source, 
		sgIdx_t*  restrict pat, 
		size_t    delta, 
		size_t    n)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 8; j++) {
            target[i*8+j] = source[pat[j]];
        }
        source += delta;    
    }
}

void gather_stride16(
		sgData_t* restrict target, 
		sgData_t* restrict source, 
		sgIdx_t*  restrict pat, 
        size_t    stride,
		size_t    delta, 
		size_t    n)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 16; j++) {
            //printf("%zu <- %zu\n", i*16+j, pat[j]);
            target[i*16+j] = source[pat[j]];
        }
        source += delta;    
    }
}

void sg_accum_omp(
            sgData_t* restrict target, 
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n)
{

#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[si[i]];
	}
}

void scatter_accum_omp(
            sgData_t* restrict target, 
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n)
{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[i];
	}
}

void gather_accum_omp(
            sgData_t* restrict target, 
            sgIdx_t*     restrict ti,
            sgData_t* restrict source,
            sgIdx_t*     restrict si,
            size_t n)
{

#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[i] += source[si[i]];
	}
}
