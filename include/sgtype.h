/** @file sgtype.h
 *  @author Patrick Lavin
 *  @brief A simple include file which can be edited to change the data type 
 */

#ifndef SGTYPE_H
#define SGTYPE_H

//defines size_t
//#include <stddef.h>

//TODO: static_assert(sizeof(cl_ulong) == sizeof(sgIdx_t);
//TODO: static_assert(sizeof(cl_double) == sizeof(sgData_t);

typedef double sgData_t;
typedef unsigned long sgIdx_t;

//#include <CL/cl.h>
//#define SGTYPE_C  cl_double  /**< The OpenCL API type used in C/C++ programs*/
//#define SGTYPE_C  double  /**< The OpenCL API type used in C/C++ programs*/
//#define SGTYPE_CL double    /**< The kernel data type corresponding to SGTYPE_C*/

//#ifndef USE_OPENCL
//	typedef unsigned long cl_ulong;
//#endif

#endif //endif SGTYPE
