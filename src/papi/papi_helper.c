#include <stdlib.h>
#include <papi.h>
#include <stdio.h>
#include "papi_helper.h"
 
const char* const papi_ctr_str[] = { "ctr0", "ctr1", "ctr2", "ctr3", 0 };

void profile_start(int EventSet) {
    papi_err(PAPI_reset(EventSet));
    papi_err(PAPI_start(EventSet));
}

void profile_stop(int EventSet, long long *val) {
    papi_err(PAPI_stop(EventSet, val));
}

void papi_err(int e) {
    if (e != PAPI_OK) {
        printf("PAPI error (%d): %s\n\n", e, PAPI_strerror(e));
        exit(1);
    }
}

