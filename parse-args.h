#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#define STRING_SIZE 100
char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];

size_t source_len;
size_t target_len;
size_t index_len;
size_t block_len;
size_t seed;

void parse_args(int argc, char **argv);

int json_flag;

enum sg_backend
{
    OPENCL,
    OPENMP,
    INVALID
};

#endif 
