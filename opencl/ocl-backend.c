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
