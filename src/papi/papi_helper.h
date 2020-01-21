#ifndef PAPI_HELPER_H
#define PAPI_HELPER_H

#define PAPI_MAX_COUNTERS 4

void profile_start(int EventSet, int lineno, char *file);
void profile_stop(int EventSet, long long *val, int lineno, char *file);
void papi_err(int e, int lineno, char* file);

#endif
