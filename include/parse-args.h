/** @file parse-args.h
 *  @author Patrick Lavin
 *  @brief Provides a function to read CLI 
 */

#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H


#define STRING_SIZE 100

#define MAX_PATTERN_LEN 32

#define PAPI_MAX_COUNTERS 4;

#include <sgtype.h>


/** @brief Supported benchmark backends
 */
enum sg_backend
{
    OPENCL, /**< OpenCL Backend */
    OPENMP, /**< OpenMP CPU Backend */
    CUDA,   /**< CUDA Backend */
    SERIAL,   /**< SERIAL Backend */
    INVALID_BACKEND /**< Used as a default backend */
};

enum sg_kernel
{
    SCATTER, 
    GATHER, 
    SG,    
    INVALID_KERNEL
};

enum sg_op
{
    OP_COPY,
    OP_ACCUM
};

enum noidx_type
{
    UNIFORM,
    MS1,
    CUSTOM,
    CONFIG_FILE,
    INVALID
};

/*
enum state
{
    NOTRUN,
    INVALID_STATE,
    VALID_STATE
}; 
*/

struct run_config
{
    spSize_t pattern_len;
    spIdx_t  pattern[MAX_PATTERN_LEN];
    ssize_t delta;
    size_t deltas[MAX_PATTERN_LEN];
    size_t deltas_ps[MAX_PATTERN_LEN];
    size_t deltas_len;
    enum sg_kernel kernel;
    enum noidx_type type;
    spSize_t generic_len;
    size_t wrap;
};

/** @brief Read command-line arguments and populate global variables. 
 *  @param argc Value passed to main
 *  @param argv Value passed to main
 */
struct run_config parse_args(int argc, char **argv);

#endif 
