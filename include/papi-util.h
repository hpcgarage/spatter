#ifndef PAPI_UTIL_H
#define PAPI_UTIL_H

//This file contains helper functions for using PAPI *if* support is compiled in
#if defined ( USE_PAPI )
#include <papi.h>

//Function to print PAPI events to a file
void dump_papi_to_file(float real_time, float proc_time);

//Include PAPI error handler
void handle_error (int retval)
{
    /* print error to stderr and exit */
    PAPI_perror(retval);
    exit(1);
}
#endif

#endif

