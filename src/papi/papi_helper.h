#ifndef PAPI_HELPER_H
#define PAPI_HELPER_H

#define PAPI_MAX_COUNTERS 4

void profile_start(int EventSet);
void profile_stop(int EventSet, long long *val);
void papi_err(int e);

#endif
