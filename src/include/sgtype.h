/** @file sgtype.h
 *  @author Patrick Lavin
 *  @brief A simple include file which can be edited to change the data type 
 */

#ifndef SGTYPE_H
#define SGTYPE_H
#include <assert.h>

#ifdef USE_OPENCL
#include "cl-helper.h"
static_assert(sizeof(cl_ulong) == sizeof(unsigned long), "Due to size differences between cl_ulong and unsigned long, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_double) == sizeof(double), "Due to size differences between cl_double and double, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_uint) == sizeof(unsigned int), "Due to size differences between cl_uint and unsigned int, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_float) == sizeof(float), "Due to size differences between cl_double and double, we cannot compile with OpenCL support on your system");
#endif

typedef double  sgData_t;
#define SGD "%lf"
typedef unsigned long sgIdx_t;
typedef unsigned long spIdx_t;
#define SGI "%lu"
typedef long sgsIdx_t;
#define SGS "%ld"

typedef size_t spSize_t;
#define SPS "%zu"

#endif //endif SGTYPE
