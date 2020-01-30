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

void read_trace(struct trace *t, const char *filename);

void print_trace(struct trace t);

void reweight_trace(struct trace t);
#endif
