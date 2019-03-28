#include "openmp_kernels.h"

#define SIMD 4

void sg_omp(
            sgData_t* restrict target,   
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
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
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{

float real_time, proc_time, mflops;
long long flpins;
int retval;

#ifdef USE_PAPI
/* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops)) == 0)
  	printf("Failed!\n");
#endif


#pragma omp parallel for simd safelen(SIMD)
	for(long i = 0; i < n; i++){
	    target[ti[i]] = source[i];
	}

#ifdef USE_PAPI
/* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops)) == 0)
  	printf("Failed!\n");
  
  //Dump PAPI stats to a file 
  dump_papi_to_file(real_time, proc_time);
#endif


}

void gather_omp(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{

float real_time, proc_time, mflops;
long long flpins;
int retval;

#ifdef USE_PAPI
/* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops)) == 0)
  	printf("Failed!\n");
#endif

//Users may want to set a specific safelen value like 32
#pragma omp parallel for simd safelen(SIMD)
#pragma prefervector
	for(long i = 0; i < n; i++){
	    target[i] = source[si[i]];
	}

#ifdef USE_PAPI
/* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops)) == 0)
  	printf("Failed!\n");

  //Dump PAPI stats to a file 
  dump_papi_to_file(real_time, proc_time);
#endif
}

void sg_accum_omp(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{

#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[si[i]];
	}
}

void scatter_accum_omp(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{
#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[ti[i]] += source[i];
	}
}

void gather_accum_omp(
            sgData_t* restrict target, 
            long*     restrict ti,
            sgData_t* restrict source,
            long*     restrict si,
            size_t n)
{

#pragma omp parallel for schedule(runtime)
	for(long i = 0; i < n; i++){
	    target[i] += source[si[i]];
	}
}
