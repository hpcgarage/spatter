#include <getopt.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <time.h>
#include "parse-args.h"
#include "backend-support-tests.h"

#define SOURCE     1000
#define TARGET     1001 
#define INDEX      1002
#define BLOCK      1003
#define SEED       1004
#define VALIDATE   1005

#define INTERACTIVE "INTERACTIVE"

extern char platform_string[STRING_SIZE];
extern char device_string[STRING_SIZE];
extern char kernel_file[STRING_SIZE];
extern char kernel_name[STRING_SIZE];

extern size_t source_len;
extern size_t target_len;
extern size_t index_len;
extern size_t block_len;
extern size_t seed;
extern size_t vector_len;
extern size_t R;
extern size_t N;
extern size_t workers;
extern int json_flag;
extern int validate_flag;
extern int print_header_flag;

extern enum sg_op op;

FILE *err_file;

void error(char *what, int code){
    if (code)
        fprintf(err_file, "Error: ");
    else
        fprintf(err_file, "Warning: ");

    fprintf(err_file, "%s", what);
    fprintf(err_file, "\n");
    if(code) exit(code);
}

void safestrcopy(char *dest, char *src){
    dest[0] = '\0';
    strncat(dest, src, STRING_SIZE-1);
}

void parse_args(int argc, char **argv)
{
    static int platform_flag = 0;
    extern enum sg_backend backend;
    extern enum sg_kernel kernel;
    source_len = 0;
    target_len = 0;
    index_len  = 0;
    block_len  = 1;
    seed       = time(NULL); 
    err_file   = stderr;

    size_t generic_len = 0;
    size_t sparsity = 1;
    int supress_errors = 0;

	static struct option long_options[] =
    {
    	/* These options set a flag. */
        {"no-print-header", no_argument, &print_header_flag, 0},
        {"nph",             no_argument, &print_header_flag, 0},
        {"backend",         required_argument, NULL, 'b'},
        {"cl-platform",     required_argument, NULL, 'p'},
        {"cl-device",       required_argument, NULL, 'd'},
        {"kernel-file",     required_argument, NULL, 'f'},
        {"kernel-name",     required_argument, NULL, 'k'},
        {"source-len",      required_argument, NULL, SOURCE},
        {"target-len",      required_argument, NULL, TARGET},
        {"index-len",       required_argument, NULL, INDEX},
        {"block-len",       required_argument, NULL, BLOCK},
        {"seed",            required_argument, NULL, SEED},
        {"vector-len",      required_argument, NULL, 'v'},
        {"generic-len",     required_argument, NULL, 'l'},
        {"runs",            required_argument, NULL, 'R'},
        {"loops",           required_argument, NULL, 'N'},
        {"workers",         required_argument, NULL, 'W'},
        {"op",              required_argument, NULL, 'o'},
        {"sparsity",        required_argument, NULL, 's'},
        {"supress-errors",  no_argument,       NULL, 'q'},
        {"validate",        no_argument, &validate_flag, 1},
        {"interactive",     no_argument,       0, 'i'},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, "W:l:k:s:qv:",
                         long_options, &option_index);

        switch(c){
            case 'b':
                if(!strcasecmp("OPENCL", optarg)){
                    backend = OPENCL;
                }
                else if(!strcasecmp("OPENMP", optarg)){
                    backend = OPENMP;
                }
                else if(!strcasecmp("CUDA", optarg)){
                    backend = CUDA;
                }
                break;
            case 'p':
                safestrcopy(platform_string, optarg);
                break;
            case 'd':
                safestrcopy(device_string, optarg);
               break;
            case 'i':
                safestrcopy(platform_string, INTERACTIVE);
                safestrcopy(device_string, INTERACTIVE);
                break;
            case 'f':
                safestrcopy(kernel_file, optarg);
                break;
            case 'k':
                safestrcopy(kernel_name, optarg);
                if (!strcasecmp("SG", optarg)) {
                    kernel = SG;
                }
                else if (!strcasecmp("SCATTER", optarg)) {
                    kernel = SCATTER;
                }
                else if (!strcasecmp("GATHER", optarg)) {
                    kernel = GATHER;
                }
                break;
            case 'o':
                if (!strcasecmp("COPY", optarg)) { 
                    op = OP_COPY;
                } else if (!strcasecmp("ACCUM", optarg)) {
                    op = OP_ACCUM;
                } else {
                    error("Unrecognzied op type", 1);
                }
                break;
            case SOURCE:
                sscanf(optarg, "%zu", &source_len);
                break;
            case TARGET:
                sscanf(optarg, "%zu", &target_len);
                break;
            case INDEX:
                sscanf(optarg, "%zu", &index_len);
                break;
            case BLOCK:
                sscanf(optarg, "%zu", &block_len);
                break;
            case SEED:
                sscanf(optarg, "%zu", &seed);
                break;
            case 'v':
                sscanf(optarg, "%zu", &vector_len);
                if (vector_len < 1) {
                    printf("Invalid vector len\n");
                    exit(1);
                }
                break;
            case 'R':
                sscanf(optarg, "%zu", &R);
                break;
            case 'N':
                sscanf(optarg, "%zu", &N);
                break;
            case 'W':
                sscanf(optarg, "%zu", &workers);
                break;
            case 'l':
                sscanf(optarg,"%zu", &generic_len);
                break;
            case 's':
                sscanf(optarg,"%zu", &sparsity);
                break;
            case 'q':
                err_file = fopen("/dev/null", "w");
                break;
            default:
                break;

        }

    }

    /* Check argument coherency */
    if(backend == INVALID_BACKEND){
        if (sg_cuda_support()) {
            backend = CUDA;
            error ("No backend specified, guessing CUDA", 0);
        }
        else if (sg_opencl_support()) {
            backend = OPENCL;
            error ("No backend specified, guessing OpenCL", 0);
        }
        else if (sg_openmp_support()) { 
            backend = OPENMP;
            error ("No backend specified, guessing OpenMP", 0);
        }
        else
        {
            error ("No backends available! Please recompile sgbench with at least one backend.", 1);
        }
    }

    //Check backend
    if(backend != INVALID_BACKEND){
        if(backend == OPENCL){
            if(platform_string[0] == '\0'){
                safestrcopy(platform_string, INTERACTIVE);
                safestrcopy(device_string, INTERACTIVE);
            }
            if(device_string[0] == '\0'){
                safestrcopy(platform_string, INTERACTIVE);
                safestrcopy(device_string, INTERACTIVE);
            }
        }
    }

    if (kernel == INVALID_KERNEL) {
        error("Kernel unspecified, guess GATHER", 0);
        kernel = GATHER;
    }

    //Check buffer lengths
    if (generic_len <= 0){

        if (source_len <= 0 && target_len <= 0 && index_len <= 0) {
            error ("Please specifiy at least one of : src_len, target_len, idx_len", 1);
        }
        if (source_len > 0 && target_len <= 0) {
            target_len = source_len;
        }
        if (source_len > 0 && index_len <= 0) {
            index_len = source_len;
        }
        if (target_len > 0 && source_len <= 0) {
            source_len = target_len;
        }
        if (target_len > 0 && index_len <= 0) {
            index_len = target_len;
        }
        if (index_len > 0 && source_len <= 0) {
            source_len = index_len;
        }
        if (index_len > 0 && target_len <= 0) {
            target_len = index_len;
        }
    }
    else{
        if (source_len > 0 || target_len > 0 || index_len > 0) {
            error ("If you specify a generic length, source_len, target_len, and index_len will be ignored.", 0);
        }

        index_len = generic_len;
        if (kernel == SCATTER) {
            target_len = generic_len * sparsity;
            source_len = generic_len;
        }
        else if (kernel == GATHER) {
            target_len = generic_len;    
            source_len = generic_len * sparsity;
        }
        else if (kernel == SG) {
            target_len = generic_len * sparsity;
            source_len = generic_len * sparsity;
        }
    }

    if(block_len < 1){
        error("Invalid index-len", 1);
    }
    if (workers < 1){
        error("Too few workers. Changing to 1.", 0);
        workers = 1;
    }

    /* Seed rand */
    srand(seed);


}
