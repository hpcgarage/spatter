#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "papi-util.h"

#if defined ( USE_PAPI )
void handle_error (int retval)
{
    char err_str[64];
    /* print error to stderr and exit */
    sprintf(err_str,"papi-util.c: handle_error(): retval = %d\n",retval);
#if USE_PAPI
    PAPI_perror(err_str);
#else
    fprintf(stderr,"%s",err_str);
#endif
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
#endif


