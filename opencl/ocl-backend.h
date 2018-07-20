/*! \file layout
 \date July 5, 2018
 \author Jeffrey Young 
 \brief Header file for OpenCL backend
*/

#ifndef OCL_BACKEND_H
#define OCL_BACKEND_H

#include "cl-helper.h"
#include "sgtype.h"
#include "sgbuf.h"

    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_mem_flags flags; 
    cl_kernel sgp;
    
    cl_event e;

void initialize_dev_ocl(char* platform_string, char* device_string);

void create_dev_buffers_ocl(sgDataBuf source, sgDataBuf target, sgIndexBuf si, sgIndexBuf ti, size_t index_len, size_t block_len, size_t worksets, size_t N);

#endif //end OCL_BACKEND
