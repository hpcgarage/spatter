#ifndef TRACE_UTIL_H
#define TRACE_UTIL_H

#include "sgtype.h"

struct instruction {
    int type;
    size_t data_type_size;
    size_t count;
    size_t length;
    double pct;
    double cpct; //cumulative pct
    sgsIdx_t *delta;
};

struct trace {
    struct instruction *in;
    size_t length;
};

int read_trace(struct trace *t, const char *filename);

int print_trace(struct trace t);

int reweight_trace(struct trace t);
#endif
