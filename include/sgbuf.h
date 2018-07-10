/** @file sgbuf.h
 *  @brief Functions to help deal with allocating and filling buffers. 
 *
 *  @author Patrick Lavin
 *
 */
#ifndef SGBUF_H
#define SGBUF_H

#include "sgtype.h"
#include "opencl/cl-helper.h"

/** @brief Describes a buffer object containing data to be scattered/gathered 
 */
typedef struct sgDataBuf_t{
    SGTYPE_C *host_ptr; /**< Points to data on the host (CPU) */
    #ifdef USE_OPENCL
    cl_mem dev_ptr;     /**< Points to data on the device if using the OCL backend */
    #endif	
    size_t len;         /**< The length of the buffers (in blocks) */
    size_t size;        /**< The size of the buffer (in bytes) */
    size_t block_len;   /**< The length of a block (number of SGTYPEs in a workset */
}sgDataBuf;

/** @brief Describes a buffer object describing how data will be scattered/gathered */
typedef struct sgIndexBuf_t{
    #ifdef USE_OPENCL
    cl_ulong *host_ptr; /**< Points to an index buffer on the host (CPU) */
    cl_mem dev_ptr;     /**< Points to an index buffer on the device if using the OCL backend */
    #else    
    unsigned long *host_ptr;
    unsigned long dev_ptr;

    #endif
    size_t len;         /**< The length of the buffer (number of cl_ulongs) */
    size_t size;        /**< The size of the buffer (in bytes) */
}sgIndexBuf;

/** @brief Fill buf with random values.
 *  @param buf Data buffer to be filled, should be pre-allocated
 *  @param len Length of buf
 */
void random_data(SGTYPE_C *buf, size_t len);

/** @brief Fill an index buffer with the indices [0:len-1] 
 *  @param idx The index buffer
 *  @param len The length of the buffer
 *  @param worksets The number of worksets
 */
void linear_indices(cl_ulong *idx, size_t len, size_t worksets);

/** @brief A helper function to create buffers on devices 
 *  @param context The OpenCL context on which the buffer will be created
 *  @param flags The flags to use in the call to clCreateBuffer
 *  @param size The size of the buffer to be allocated
 *  @param host_ptr The location from which to copy the data from, if not null and the proper flags are set. 
 */

#ifdef USE_OPENCL
cl_mem clCreateBufferSafe(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr);
#endif

#endif
