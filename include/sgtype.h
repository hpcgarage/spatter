/** @file sgtype.h
 *  @author Patrick Lavin
 *  @brief A simple include file which can be edited to change the data type 
 */

#ifndef SGTYPE_H
#define SGTYPE_H
#include <assert.h>

#ifndef SG_DATA_TYPE
    #define SG_DATA_TYPE double
#endif
#ifndef SG_INDEX_TYPE
    #define SG_INDEX_TYPE unsigned long
#endif

#ifdef USE_OPENCL
#include "cl-helper.h"
static_assert(sizeof(cl_ulong) == sizeof(unsigned long), "Due to size differences between cl_ulong and unsigned long, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_double) == sizeof(double), "Due to size differences between cl_double and double, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_uint) == sizeof(unsigned int), "Due to size differences between cl_uint and unsigned int, we cannot compile with OpenCL support on your system");
static_assert(sizeof(cl_float) == sizeof(float), "Due to size differences between cl_double and double, we cannot compile with OpenCL support on your system");
#endif

typedef SG_DATA_TYPE  sgData_t;
typedef SG_INDEX_TYPE sgIdx_t;

#endif //endif SGTYPE
