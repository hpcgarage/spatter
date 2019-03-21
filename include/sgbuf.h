/** @file sgbuf.h
 *  @brief Functions to help deal with allocating and filling buffers. 
 *
 *  @author Patrick Lavin
 *
 */
#ifndef SGBUF_H
#define SGBUF_H

#include <stdlib.h> //rand()
#include <stddef.h> //size_t
#include "sgtype.h"
#include "trace-util.h"
#ifdef USE_OPENCL
#include "../opencl/cl-helper.h"
#endif

/** @brief Describes a buffer object containing data to be scattered/gathered 
 */
typedef struct sgDataBuf_t{
    sgData_t *host_ptr;        /**< Points to data on the host (CPU) */
    #ifdef USE_OPENCL
    cl_mem dev_ptr_opencl;     /**< Points to data on the OpenCL device */ 
    #endif
    #ifdef USE_CUDA
    sgData_t *dev_ptr_cuda;
    #endif    
    size_t len;         /**< The length of the buffers (in blocks) */
    size_t size;        /**< The size of the buffer (in bytes) */
}sgDataBuf;

/** @brief Describes a buffer object describing how data will be scattered/gathered */
typedef struct sgIndexBuf_t{
    sgIdx_t *host_ptr;    /**< Points to an index buffer on the host (CPU) */

    #ifdef USE_CUDA
    sgIdx_t *dev_ptr_cuda;/**< Points to an index buffer on the CUDA device */
    #endif

    #ifdef USE_OPENCL
    cl_mem dev_ptr_opencl;/**< Points to an index buffer on the OpenCL device */
    #endif

    size_t len;           /**< The length of the buffer (number of sgIdx_t's) */
    size_t size;          /**< The size of the buffer (in bytes) */
    size_t stride;
}sgIndexBuf;

/** @brief Fill buf with random values.
 *  @param buf Data buffer to be filled, should be pre-allocated
 *  @param len Length of buf
 */
void random_data(sgData_t *buf, size_t len);

/** @brief Fill an index buffer with the indices [0:len-1] 
 *  @param idx The index buffer
 *  @param len The length of the buffer
 *  @param worksets The number of worksets
 *  @param gap The stride of the indices
 */
void linear_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t stride, int randomize);

void wrap_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t stride, size_t wrap);
/** @brief A helper function to create buffers on devices 
 *  @param context The OpenCL context on which the buffer will be created
 *  @param flags The flags to use in the call to clCreateBuffer
 *  @param size The size of the buffer to be allocated
 *  @param host_ptr The location from which to copy the data from, if not null and the proper flags are set. 
 */

void ms1_indices(sgIdx_t *idx, size_t len, size_t worksets, size_t run, size_t gap);
size_t trace_indices( sgIdx_t *idx, size_t len, struct trace tr);
#ifdef USE_OPENCL
//TODO: why is it a void*? 
cl_mem clCreateBufferSafe(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr);
#elif defined USE_CUDA
//double* cudaCreateBufferSafe(
#endif

#endif
