/** @file parse-args.h
 *  @author Patrick Lavin
 *  @brief Provides a function to read CLI
 */

#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#define WARN 0
#define ERROR 1


#define STRING_SIZE 10000
#define MAX_PATTERN_LEN 1024

#include <sgtype.h>
#include <stdint.h>

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
    INVALID_KERNEL=0,
    SCATTER,
    GATHER,
    SG,
};

enum sg_op
{
    OP_COPY,
    OP_ACCUM,
    INVALID_OP
};

//Specifies the indexing or offset type
enum idx_type
{
    UNIFORM,
    MS1,
    LAPLACIAN,
    CUSTOM,
    CONFIG_FILE,
    HYDRO,
    INVALID_IDX
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
    // keep arrays at top so they are aligned
    spIdx_t  pattern[MAX_PATTERN_LEN];
    size_t deltas[MAX_PATTERN_LEN];
    size_t deltas_ps[MAX_PATTERN_LEN];
    spSize_t pattern_len;
    ssize_t delta;
    size_t deltas_len;
    enum sg_kernel kernel;
    enum idx_type type;
    spSize_t generic_len;
    size_t wrap;
    size_t nruns;
    char pattern_file[STRING_SIZE];
    char generator[STRING_SIZE];
    char name[STRING_SIZE];
    size_t random_seed;
    size_t omp_threads;
    enum sg_op op;
    size_t vector_len;
    unsigned int shmem;
    size_t local_work_size;
    double *time_ms;
    long long **papi_ctr;
    int papi_counters;
    int morton;
    uint32_t *morton_order;
    uint32_t *morton_order_dev;
};

struct backend_config
{
    enum sg_backend backend;
    enum sg_kernel kernel;
    enum sg_op op;

    char platform_string[STRING_SIZE];
    char device_string[STRING_SIZE];
    char kernel_file[STRING_SIZE];
    char kernel_name[STRING_SIZE];

};

/** @brief Read command-line arguments and populate global variables.
 *  @param argc Value passed to main
 *  @param argv Value passed to main
 */
void parse_args(int argc, char **argv, int *nrc, struct run_config **rc);
struct run_config parse_runs(int arrr, char **argv);
void error (char* what, int code);
void print_run_config(struct run_config rc);
#endif
