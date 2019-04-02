#include <getopt.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "parse-args.h"
#include "backend-support-tests.h"

#ifdef USE_CUDA 
#include "cuda-backend.h"
#endif

#define SOURCE      1000
#define TARGET      1001 
#define INDEX       1002
#define BLOCK       1003
#define SEED        1004
#define VALIDATE    1005
#define MS1_PATTERN 1006
#define MS1_GAP     1007
#define MS1_RUN     1008

#define INTERACTIVE "INTERACTIVE"

extern char platform_string[STRING_SIZE];
extern char device_string[STRING_SIZE];
extern char kernel_file[STRING_SIZE];
extern char kernel_name[STRING_SIZE];
extern char config_file[STRING_SIZE];

extern size_t source_len;
extern size_t target_len;
extern size_t index_len;
extern size_t wrap;
extern size_t seed;
extern size_t vector_len;
extern size_t R;
extern size_t local_work_size;
extern size_t workers;
extern size_t ms1_gap;
extern size_t ms1_run;
extern int ms1_flag;
extern int config_flag;
extern int json_flag;
extern int validate_flag;
extern int print_header_flag;
extern int random_flag;
extern unsigned int shmem;
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
    seed       = time(NULL); 
    err_file   = stderr;

    safestrcopy(platform_string, "NONE");
    safestrcopy(device_string,   "NONE");
    safestrcopy(kernel_file,     "NONE");
    safestrcopy(kernel_name,     "NONE");

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
        {"seed",            required_argument, NULL, SEED},
        {"vector-len",      required_argument, NULL, 'v'},
        {"generic-len",     required_argument, NULL, 'l'},
        {"runs",            required_argument, NULL, 'R'},
        {"workers",         required_argument, NULL, 'W'},
        {"wrap",            required_argument, NULL, 'w'},
        {"op",              required_argument, NULL, 'o'},
        {"uniform-stride",  required_argument, NULL, 's'},
        {"local-work-size", required_argument, NULL, 'z'},
        {"shared-mem",      required_argument, NULL, 'm'},
        {"ms1-pattern",     no_argument,       NULL, MS1_PATTERN},
        {"ms1-gap",         required_argument, NULL, MS1_GAP},
        {"ms1-run",         required_argument, NULL, MS1_RUN},
        {"config-file",     required_argument, NULL, 't'},
        {"supress-errors",  no_argument,       NULL, 'q'},
        {"random",          no_argument,       NULL, 'y'},
        {"validate",        no_argument, &validate_flag, 1},
        {"interactive",     no_argument,       0, 'i'},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, "W:l:k:s:qv:R:p:d:f:b:z:m:yw:",
                         long_options, &option_index);

        switch(c){
            case 'b':
                if(!strcasecmp("OPENCL", optarg)){
                    if (!sg_opencl_support()) {
                        error("You did not compile with support for OpenCL", 1);
                    }
                    backend = OPENCL;
                }
                else if(!strcasecmp("OPENMP", optarg)){
                    if (!sg_openmp_support()) {
                        error("You did not compile with support for OpenMP", 1);
                    }
                    backend = OPENMP;
                }
                else if(!strcasecmp("CUDA", optarg)){
                    if (!sg_cuda_support()) {
                        error("You did not compile with support for CUDA", 1);
                    }
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
            case 'W':
                sscanf(optarg, "%zu", &workers);
                break;
            case 'w':
                sscanf(optarg, "%zu", &wrap);
                break;
            case 'l':
                sscanf(optarg,"%zu", &generic_len);
                break;
            case 's':
                sscanf(optarg,"%zu", &sparsity);
                break;
            case 'z':
                sscanf(optarg,"%zu", &local_work_size);
                break;
            case 'y':
                random_flag = 1;
                break;
            case 'm':
                sscanf(optarg,"%u", &shmem);
                break;
            case 'q':
                err_file = fopen("/dev/null", "w");
                break;
            case MS1_PATTERN:
                ms1_flag = 1;
                break;
            case MS1_RUN:
                sscanf(optarg, "%zu", &ms1_run);
                break;
            case MS1_GAP:
                sscanf(optarg, "%zu", &ms1_gap);
                break;
            case 't':
                safestrcopy(config_file, optarg);
                config_flag = 1;
                break;
            default:
                break;

        }

    }

    if (generic_len <= 0) {
        error ("Length not specified. Default is 16 (elements)", 0);
        generic_len = 16;
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

    if(backend == OPENCL){
        if(!strcasecmp(platform_string, "NONE")){
            safestrcopy(platform_string, INTERACTIVE);
            safestrcopy(device_string, INTERACTIVE);
        }
        if(!strcasecmp(device_string, "NONE")){
            safestrcopy(platform_string, INTERACTIVE);
            safestrcopy(device_string, INTERACTIVE);
        }
    }

    #ifdef USE_CUDA
    if (backend == CUDA) {
        int dev = find_device_cuda(device_string);
        if (dev == -1) {
            error("Specified CUDA device not found or no device specified. Using device 0", 0);
            dev = 0;
        }
        cudaSetDevice(dev);
    }
    #endif

    if (kernel == INVALID_KERNEL) {
        error("Kernel unspecified, guess GATHER", 0);
        kernel = GATHER;
        safestrcopy(kernel_name, "gather");
    }

    if (kernel == SCATTER) {
        sprintf(kernel_name, "%s%zu", "scatter", vector_len);
    } else if (kernel == GATHER) {
        sprintf(kernel_name, "%s%zu", "gather", vector_len);
    } else if (kernel == SG) {
        sprintf(kernel_name, "%s%zu", "sg", vector_len);
    }

    if (!strcasecmp(kernel_file, "NONE")) {
        error("Kernel file unspecified, guessing kernels/kernels_vector.cl", 0);
        safestrcopy(kernel_file, "kernels/kernels_vector.cl");
    }

    //Check buffer lengths
    if (ms1_flag) {
        if (kernel == SCATTER) {
            source_len = generic_len;
            target_len = (generic_len / ms1_run) * (ms1_run + ms1_gap);
            index_len = generic_len;
        } else if (kernel == GATHER) {
            target_len = generic_len;
            source_len = (generic_len / ms1_run) * (ms1_run + ms1_gap);
            index_len = generic_len;
        }
    }
    else{
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

    if (workers < 1){
        error("Too few workers. Changing to 1.", 0);
        workers = 1;
    }
    
    if(ms1_flag) {
        assert(ms1_run > 0);
        assert(ms1_gap > 0);
    }

    /* Seed rand */
    srand(seed);


}
