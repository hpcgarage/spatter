#ifndef PAPI_UTIL_H
#define PAPI_UTIL_H

//This file contains helper functions for using PAPI *if* support is compiled in
#if defined ( USE_PAPI )
#include <papi.h>

//Struct to hold PAPI events and counters
struct papi_t{
  int num;
  int *events;
  long long int *counters;
};

void print_papi_stats(struct papi_t *papi);

//Function to print PAPI events to a file
void dump_papi_to_file(struct papi_t *papi, FILE *papiFile);

//Include PAPI error handler
void handle_error (int retval);

void papi_struct_set(struct papi_t *papi, int num_events, int *events, long long int *counters);

#endif

#endif

