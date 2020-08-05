#include "backend-support-tests.h"

int sg_sycl_support()
{
#if defined USE_SYCL
    return 1;
#else
    return 0;
#endif
}

int sg_cuda_support()
{
#if defined USE_CUDA
    return 1;
#else 
    return 0;
#endif
}

int sg_opencl_support()
{
#if defined USE_OPENCL
    return 1;
#else 
    return 0;
#endif
}

int sg_openmp_support()
{
#if defined USE_OPENMP
    return 1;
#else 
    return 0;
#endif
}

int sg_serial_support()
{
#if defined USE_SERIAL
    return 1;
#else 
    return 0;
#endif
}
