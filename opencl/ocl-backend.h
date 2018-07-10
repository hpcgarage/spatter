/*! \file layout
 \date July 5, 2018
 \author Jeffrey Young 
 \brief Header file for OpenCL backend
*/

#ifndef OCL_BACKEND_H
#define OCL_BACKEND_H

#include "opencl/cl-helper.h"

    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_mem_flags flags; 
    cl_kernel sgp;
    
    cl_event e;

void initialize_dev_ocl(char* platform_string, char* device_string);

#endif //end OCL_BACKEND
