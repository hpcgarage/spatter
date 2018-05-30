#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#define STRING_SIZE 100

void parse_args(int argc, char **argv);

enum sg_backend
{
    OPENCL,
    OPENMP,
    INVALID
};

#endif 
