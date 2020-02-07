#include <stdlib.h>
#include <papi.h>
#include <stdio.h>
#include "papi_helper.h"

const char* const papi_ctr_str[] = { "ctr0", "ctr1", "ctr2", "ctr3", 0 };

void profile_start(int EventSet, int lineno, char *file) {
    papi_err(PAPI_reset(EventSet), lineno, file);
    papi_err(PAPI_start(EventSet), lineno, file);
}

void profile_stop(int EventSet, long long *val, int lineno, char *file) {
    papi_err(PAPI_stop(EventSet, val), lineno, file);
}

void papi_err(int e, int lineno, char *file) {
    if (e != PAPI_OK) {
        printf("PAPI error (%d): %s [%s:%d]\n\n", e, PAPI_strerror(e), file, lineno);
        exit(1);
    }
}

