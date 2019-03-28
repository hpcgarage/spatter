#ifndef UTIL_H
#define UTIL_H



void make_upper (char* s);

void print_data (sgData_t *buf, size_t len);

void dump_papi_to_file (float real_time, float proc_time);

//Include PAPI error handler

#if defined ( USE_PAPI )
    #include <papi.h>
void handle_error (int retval)
{
    /* print error to stderr and exit */
    PAPI_perror(retval);
    exit(1);
}
#endif

#endif

