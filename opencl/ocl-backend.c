/*! \file ocl-backend
 \date July 5, 2018
 \author Jeffrey Young
 \brief Source file for the OpenCL backend
 */

#include "ocl-backend.h"

void initialize_dev_ocl(char* platform_string, char* device_string)
{
	create_context_on(platform_string, device_string, 0, 
                      &context, &queue, &device, 1);

}

void create_dev_buffers_ocl(sgDataBuf *source, sgDataBuf *target, sgIndexBuf *si, sgIndexBuf *ti, size_t block_len)
{

        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
        source->dev_ptr = clCreateBufferSafe(context, flags, source->size, source->host_ptr);
        si->dev_ptr = clCreateBufferSafe(context, flags, si->size, si->host_ptr);
        ti->dev_ptr = clCreateBufferSafe(context, flags, ti->size, ti->host_ptr);

        flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
        target->dev_ptr = clCreateBufferSafe(context, flags, target->size, NULL); 

}
