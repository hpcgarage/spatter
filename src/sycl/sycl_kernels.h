#ifndef SG_SYCL_H
#define SG_SYCL_H
#include <sycl/sycl.hpp>
__attribute__((always_inline)) void my_kernel();
__attribute__((always_inline)) void scatter(double* target, double* source, long* ti, long* si);
#endif
