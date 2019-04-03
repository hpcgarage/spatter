#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "papi-util.h"

void handle_error (int retval)
{
    /* print error to stderr and exit */
    PAPI_perror(retval);
    exit(1);
}

void papi_struct_set(struct papi_t *papi, int num_events, int *events, long long int *counters)
{
  //papi->num = num_events;
  //papi->events = events;
  //papi->counters = counters;
}

void print_papi_stats(struct papi_t *papi)
{
  for(int i = 0; i < (*papi).num; i++){
    printf("%lld ", (*papi).counters[i]);
  }
}

void dump_papi_to_file(struct papi_t *papi, FILE *filePtr)
{
  for(int i = 0; i < (*papi).num; i++)
  {
    fprintf(filePtr, "%lld ", (*papi).counters[i]);
  }
  fprintf(filePtr,"\n");
}
