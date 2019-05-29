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
#define USTRIDE     1009
#define CLPLATFORM  1010
#define CLDEVICE    1011

#define INTERACTIVE "INTERACTIVE"


extern char platform_string[STRING_SIZE];
extern char device_string[STRING_SIZE];
extern char kernel_file[STRING_SIZE];
extern char kernel_name[STRING_SIZE];
extern char config_file[STRING_SIZE];

extern size_t source_len;
extern size_t target_len;
extern size_t index_len;
extern size_t generic_len;
extern size_t wrap;
extern size_t seed;
extern size_t vector_len;
extern size_t R;
extern size_t local_work_size;
extern size_t workers;
extern size_t ms1_gap;
extern size_t ms1_run;
extern ssize_t us_stride;
extern ssize_t us_delta;
extern int ms1_flag;
extern int config_flag;
extern int json_flag;
extern int validate_flag;
extern int print_header_flag;
extern int random_flag;
extern int ustride_flag;
extern unsigned int shmem;
extern enum sg_op op;

extern int noidx_flag;
extern int noidx_explicit_mode;
extern int noidx_predef_us_mode;
extern int noidx_predef_ms1_mode;
extern int noidx_file_mode;

extern size_t noidx_pattern[MAX_PATTERN_LEN];
extern size_t noidx_pattern_len;
extern char   noidx_pattern_file[STRING_SIZE];

extern ssize_t noidx_delta;
extern ssize_t noidx_us_stride;
extern size_t noidx_ms1_deltas[MAX_PATTERN_LEN];
extern size_t noidx_ms1_breaks[MAX_PATTERN_LEN];
extern size_t noidx_ms1_deltas_len;
extern size_t noidx_ms1_breaks_len;
extern ssize_t noidx_ms1_delta;


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

    size_t sparsity = 1;
    int supress_errors = 0;

	static struct option long_options[] =
    {
    	/* These options set a flag. */
        {"no-print-header", no_argument, &print_header_flag, 0},
        {"nph",             no_argument, &print_header_flag, 0},
        {"backend",         required_argument, NULL, 'b'},
        {"cl-platform",     required_argument, NULL, CLPLATFORM},
        {"cl-device",       required_argument, NULL, CLDEVICE},
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
        {"ustride",         required_argument, NULL, USTRIDE},
        {"config-file",     required_argument, NULL, 't'},
        {"pattern",         required_argument, NULL, 'p'},
        {"delta",           required_argument, NULL, 'p'},
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
                else if(!strcasecmp("SERIAL", optarg)){
                    if (!sg_serial_support()) {
                        error("You did not compile with support for serial execution", 1);
                    }
                    backend = SERIAL;
                }
                break;
            case CLPLATFORM:
                safestrcopy(platform_string, optarg);
                break;
            case CLDEVICE:
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
            case USTRIDE:
                ustride_flag = 1;
                sscanf(optarg, "%zu,%zu", &us_stride, &us_delta);
                break;
            case 'p':
                ;
                // This code will parse the arguments for NOIDX mode. 
                // Correctness checking is done after all arguments are parsed, below
                noidx_flag = 1;
                char *arg = 0;
                if ((arg=strchr(optarg, ':'))) {

                    *arg = '\0';
                    arg++; //arg now points to arguments to the pattern type

                    // FILE mode indicates that we will load a 
                    // config from a file
                    if (strstr(optarg, "FILE")) {
                        noidx_file_mode = 1;
                        safestrcopy(noidx_pattern_file, arg);
                    }

                    // Parse Uniform Stride Arguments, which are 
                    // UNIFORM:index_length:stride
                    else if (!strcmp(optarg, "UNIFORM")) {

                        noidx_predef_us_mode = 1;
                        
                        // Read the length
                        char *len = strtok(arg,":");
                        if (!len) error("UNIFORM: Index Length not found", 1);
                        if (sscanf(len, "%zd", &noidx_pattern_len) < 1)
                            error("UNIFORM: Length not parsed", 1);
                            
                        // Read the stride
                        char *stride = strtok(NULL, ":");
                        if (!stride) error("UNIFORM: Stride not found", 1);
                        if (sscanf(stride, "%zd", &noidx_us_stride) < 1)
                            error("UNIFORM: Stride not parsed", 1);

                    }

                    // Mostly Stride 1 Mode
                    // Arguments: index_length:list_of_breaks:list_of_deltas 
                    // list_of_deltas should be length 1 or the same length as 
                    // list_of_breaks.
                    // The elements of both lists should be nonnegative and 
                    // the the elements of list_of_breaks should be strictly less 
                    // than index_length
                    else if (!strcmp(optarg, "MS1")) {

                        noidx_predef_ms1_mode = 1;

                        char *len = strtok(arg,":");
                        char *breaks = strtok(NULL,":");
                        char *gaps = strtok(NULL,":");
                        
                        // Parse index length 
                        sscanf(len, "%zd", &noidx_pattern_len);

                        // Parse breaks
                        char *ptr = strtok(breaks, ",");
                        size_t read = 0;
                        if (!ptr) {
                            error ("MS1: Breaks missing", 1);
                        }            
                        if (sscanf(ptr, "%zu", &(noidx_ms1_breaks[read++])) < 1) {
                            error ("MS1: Failed to parse first break", 1);
                        }

                        while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN) {
                            if (sscanf(ptr, "%zu", &(noidx_ms1_breaks[read++])) < 1) {
                                error ("MS1: Failed to parse breaks", 1);
                            }
                        }
                        
                        noidx_ms1_breaks_len = read;

                        // Parse deltas
                        ptr = strtok(gaps, ",");
                        read = 0;
                        if (ptr) {
                            if (sscanf(ptr, "%zu", &(noidx_ms1_deltas[read++])) < 1) {
                                error ("Failed to parse first delta", 1);
                            }

                            while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN) {
                                if (sscanf(ptr, "%zu", &(noidx_ms1_deltas[read++])) < 1) {
                                    error ("Failed to parse deltas", 1);
                                }
                            }
                        }
                        else {
                            error("MS1: deltas missing",1);
                        }

                        noidx_ms1_deltas_len = read;
                    }
                    else {
                        error("Unrecognized mode in -p argument", 1);
                    }
                }
                
                // EXPLICIT mode means that the user supplied a single index buffer on the command line
                else {
                    noidx_explicit_mode = 1;
                    char *delim = ",";
                    char *ptr = strtok(optarg, delim);
                    size_t read = 0;
                    if (!ptr) {
                        error ("Pattern not found", 1);
                    }            

                    if (sscanf(ptr, "%zu", &(noidx_pattern[read++])) < 1) {
                        error ("Failed to parse first pattern element", 1);
                    }

                    while ((ptr = strtok(NULL, delim)) && read < MAX_PATTERN_LEN) {
                        if (sscanf(ptr, "%zu", &(noidx_pattern[read++])) < 1) {
                            error ("Failed to parse pattern", 1);
                        }
                    }

                    noidx_pattern_len = read;
                }
                exit(1); 
                break;
            case 'd':
                sscanf(optarg, "%zu", &us_delta);
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
        else if (sg_serial_support()) { 
            backend = SERIAL;
            error ("No backend specified, guessing Serial", 0);
        }
        else
        {
            error ("No backends available! Please recompile spatter with at least one backend.", 1);
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
    else if (ustride_flag) {
        if (kernel == GATHER) {
            assert(us_stride >= 0);
            
            index_len = 1;
            target_len = generic_len * 16; 

            ssize_t window = 16 * us_stride;
            assert(delta >= 0);
            source_len = window + (generic_len-1)*(us_delta);
            //source_len = window + (generic_len-1) * ( window + us_delta );
        } else {
            printf("Not supported yet\n");
            exit(1);
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
