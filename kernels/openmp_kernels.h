#ifndef OMP_KERNELS_H
#define OMP_KERNELS_H
void sg_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
void scatter_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
void gather_omp(double* restrict target, 
            long*   restrict ti,
            double* restrict source,
            long*   restrict si,
            long ts, 
            long ss, 
            long n, 
            long ws,
            long R,
            long B);
#endif
