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

#define SEED        1004
#define VALIDATE    1005
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
extern int config_flag;
extern int json_flag;
extern int validate_flag;
extern int print_header_flag;
extern int random_flag;
extern unsigned int shmem;
extern enum sg_op op;

extern int noidx_flag;
extern int noidx_explicit_mode;
extern int noidx_predef_us_mode;
extern int noidx_predef_ms1_mode;
extern int noidx_file_mode;
extern int noidx_onesided;

size_t noidx_pattern[MAX_PATTERN_LEN];
size_t noidx_pattern_len;
extern char   noidx_pattern_file[STRING_SIZE];

ssize_t noidx_us_stride;
size_t noidx_ms1_deltas[MAX_PATTERN_LEN];
size_t noidx_ms1_breaks[MAX_PATTERN_LEN];
size_t noidx_ms1_deltas_len;
size_t noidx_ms1_breaks_len;
ssize_t noidx_ms1_delta;

extern int verbose;

FILE *err_file;

void error(char *what, int code){
    if (code) {
        fprintf(err_file, "Error: ");
    }
    else {
        if (verbose)
            fprintf(err_file, "Warning: ");
    }

    if (verbose || code) {
        fprintf(err_file, "%s", what);
        fprintf(err_file, "\n");
    }
    if(code) exit(code);
}

void safestrcopy(char *dest, char *src){
    dest[0] = '\0';
    strncat(dest, src, STRING_SIZE-1);
}

ssize_t setincludes(size_t s, size_t* noidx_ms1_breaks, size_t noidx_ms1_breaks_len){ 
    for (size_t i = 0; i < noidx_ms1_breaks_len; i++) {
        if (noidx_ms1_breaks[i] == s) {
            return i;
        }
    }
    return -1;
}
void parse_p(char*, struct run_config *);
void print_run_config(struct run_config rc);
//void parse_args(int argc, char **argv, struct run_config *rc, int nconfigs)
struct run_config parse_args(int argc, char **argv)
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

    struct run_config rc = {0};
    rc.delta = -1;

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
        {"config-file",     required_argument, NULL, 't'},
        {"pattern",         required_argument, NULL, 'p'},
        {"delta",           required_argument, NULL, 'd'},
        {"supress-errors",  no_argument,       NULL, 'q'},
        {"random",          no_argument,       NULL, 'y'},
        {"verbose",         no_argument,       &verbose, 1},
        {"validate",        no_argument, &validate_flag, 1},
        {"interactive",     no_argument,       0, 'i'},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, "W:l:k:s:qv:R:p:d:D:f:b:z:m:yw:",
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
                    rc.kernel=SG;
                }
                else if (!strcasecmp("SCATTER", optarg)) {
                    kernel = SCATTER;
                    rc.kernel=SCATTER;
                }
                else if (!strcasecmp("GATHER", optarg)) {
                    kernel = GATHER;
                    rc.kernel=GATHER;
                }
                else {
                    error("Invalid kernel", 1);
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
            case SEED:
                sscanf(optarg, "%zu", &seed);
                break;
            case 'v':
                sscanf(optarg, "%zu", &vector_len);
                if (vector_len < 1) {
                    error("Invalid vector len", 1);
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
                sscanf(optarg, "%d", &noidx_onesided);
                printf("noidx_onesided: %d\n", noidx_onesided);
                sscanf(optarg,"%zu", &rc.wrap);
                break;
            case 'l':
                sscanf(optarg,"%zu", &generic_len);
                sscanf(optarg,"%zu", &(rc.generic_len));
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
            case 'p':
                // This code will parse the arguments for NOIDX mode. 
                // Correctness checking is done after all arguments are parsed, below
                {
                noidx_flag = 1;

                //{
                char *optarg_copy = (char*)malloc(strlen(optarg)+1);
                strcpy(optarg_copy, optarg);
                parse_p(optarg_copy, &rc);
                
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

                //}
                break;
            }
            case 'd':
                sscanf(optarg, "%zu", &(rc.delta));
                break;
            case 'D':
                {
                char *delim = ",";
                char *ptr = strtok(optarg, delim);
                size_t read = 0;
                if (!ptr) {
                    error ("Pattern not found", 1);
                }            

                if (sscanf(ptr, "%zu", &(rc.deltas[read++])) < 1) {
                    error ("Failed to parse first pattern element", 1);
                }

                while ((ptr = strtok(NULL, delim)) && read < MAX_PATTERN_LEN) {
                    if (sscanf(ptr, "%zu", &(rc.deltas[read++])) < 1) {
                        error ("Failed to parse pattern", 1);
                    }
                }
                rc.deltas_len = read;

                // rotate
                for (size_t i = 0; i < rc.deltas_len; i++) {
                    rc.deltas_ps[i] = rc.deltas[((i-1)+rc.deltas_len)%rc.deltas_len];
                    printf("rc.deltas_ps[%zu] = %zu\n",i, rc.deltas_ps[i]);
                }
                // compute prefix-sum
                
#pragma novector // ALERT: Do not remove this pragma - the cray compiler will mistakenly vectorize this loop 
                for (size_t i = 1; i < rc.deltas_len; i++) {
                    rc.deltas_ps[i] += rc.deltas_ps[i-1];
                }
                // compute max
                size_t m = rc.deltas_ps[0];
                for (size_t i = 1; i < rc.deltas_len; i++) {
                    if (rc.deltas_ps[i] > m) {
                        m = rc.deltas_ps[i];
                    }
                }
                rc.delta = m;

                break;
                }
            case 't':
                safestrcopy(config_file, optarg);
                config_flag = 1;
                break;
            default:
                break;

        }

    }

    if (generic_len <= 0) {
        error ("Length not specified. Default is 32 (elements)", 0);
        generic_len = 32;
        rc.generic_len = 32;
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
        rc.kernel = GATHER;
        safestrcopy(kernel_name, "gather");
    }

    if (kernel == SCATTER) {
        sprintf(kernel_name, "%s%zu", "scatter", vector_len);
    } else if (kernel == GATHER) {
        sprintf(kernel_name, "%s%zu", "gather", vector_len);
    } else if (kernel == SG) {
        sprintf(kernel_name, "%s%zu", "sg", vector_len);
    }

    if (!strcasecmp(kernel_file, "NONE") && backend == OPENCL) {
        error("Kernel file unspecified, guessing kernels/kernels_vector.cl", 0);
        safestrcopy(kernel_file, "kernels/kernels_vector.cl");
    }

    //Check buffer lengths

    if (noidx_flag) {
        if (noidx_explicit_mode) {
        }else if (noidx_predef_us_mode) {   
            for (int i = 0; i < noidx_pattern_len; i++) {
                noidx_pattern[i] = i*noidx_us_stride;
            }
        }
        else if (noidx_predef_ms1_mode) {
            size_t last = 0;
            ssize_t j;
            for (int i = 1; i < noidx_pattern_len; i++) {
                if ((j=setincludes(i, noidx_ms1_breaks, noidx_ms1_breaks_len))!=-1) {
                   noidx_pattern[i] = last+noidx_ms1_deltas[noidx_ms1_deltas_len>1?j:0];
                } else {
                    noidx_pattern[i] = last + 1;
                }
                last = noidx_pattern[i];
            }
        }     
        
        if (rc.delta == -1) {
            error("delta not specified, default is 8\n", 0);
            rc.delta = 8;
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
    
    /* Seed rand */
    srand(seed);

    print_run_config(rc);
    return rc;

}


void parse_p(char* optarg, struct run_config *rc) {

    char *arg = 0;
    if ((arg=strchr(optarg, ':'))) {

        *arg = '\0';
        arg++; //arg now points to arguments to the pattern type

        // FILE mode indicates that we will load a 
        // config from a file
        if (!strcmp(optarg, "FILE")) {
            //TODO
            //safestrcopy(noidx_pattern_file, arg);
            rc->type = CONFIG_FILE;
        }

        // Parse Uniform Stride Arguments, which are 
        // UNIFORM:index_length:stride
        else if (!strcmp(optarg, "UNIFORM")) {

            rc->type = UNIFORM;
            
            // Read the length
            char *len = strtok(arg,":");
            if (!len) error("UNIFORM: Index Length not found", 1);
            if (sscanf(len, "%zd", &(rc->pattern_len)) < 1)
                error("UNIFORM: Length not parsed", 1);
                
            // Read the stride
            char *stride = strtok(NULL, ":");
            ssize_t strideval = 0;
            if (!stride) error("UNIFORM: Stride not found", 1);
            if (sscanf(stride, "%zd", &strideval) < 1)
                error("UNIFORM: Stride not parsed", 1);

            for (int i = 0; i < rc->pattern_len; i++) {
                rc->pattern[i] = i*strideval;
            }

        }

        // Mostly Stride 1 Mode
        // Arguments: index_length:list_of_breaks:list_of_deltas 
        // list_of_deltas should be length 1 or the same length as 
        // list_of_breaks.
        // The elements of both lists should be nonnegative and 
        // the the elements of list_of_breaks should be strictly less 
        // than index_length
        else if (!strcmp(optarg, "MS1")) {

            rc->type = MS1;

            char *len = strtok(arg,":");
            char *breaks = strtok(NULL,":");
            char *gaps = strtok(NULL,":");

            size_t ms1_breaks[MAX_PATTERN_LEN];
            size_t ms1_deltas[MAX_PATTERN_LEN];
            size_t ms1_breaks_len = 0;
            size_t ms1_deltas_len = 0;
            
            // Parse index length 
            sscanf(len, "%zd", &(rc->pattern_len));

            // Parse breaks
            char *ptr = strtok(breaks, ",");
            size_t read = 0;
            if (!ptr) {
                error ("MS1: Breaks missing", 1);
            }            
            if (sscanf(ptr, "%zu", &(ms1_breaks[read++])) < 1) {
                error ("MS1: Failed to parse first break", 1);
            }

            while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN) {
                if (sscanf(ptr, "%zu", &(ms1_breaks[read++])) < 1) {
                    error ("MS1: Failed to parse breaks", 1);
                }
            }
             
            ms1_breaks_len = read;

            // Parse deltas
            ptr = strtok(gaps, ",");
            read = 0;
            if (ptr) {
                if (sscanf(ptr, "%zu", &(ms1_deltas[read++])) < 1) {
                    error ("Failed to parse first delta", 1);
                }

                while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN) {
                    if (sscanf(ptr, "%zu", &(ms1_deltas[read++])) < 1) {
                        error ("Failed to parse deltas", 1);
                    }
                }
            }
            else {
                error("MS1: deltas missing",1);
            }

            ms1_deltas_len = read;

            rc->pattern[0] = 0;
            size_t last = 0;
            ssize_t j;
            for (int i = 1; i < rc->pattern_len; i++) {
                if ((j=setincludes(i, ms1_breaks, ms1_breaks_len))!=-1) {
                   rc->pattern[i] = last+ms1_deltas[ms1_deltas_len>1?j:0];
                } else {
                    rc->pattern[i] = last + 1;
                }
                last = rc->pattern[i];
            }
        }
        else {
            error("Unrecognized mode in -p argument", 1);
        }
    }
    
    // CUSTOM mode means that the user supplied a single index buffer on the command line
    else {
        rc->type = CUSTOM;
        char *delim = ",";
        char *ptr = strtok(optarg, delim);
        size_t read = 0;
        if (!ptr) {
            error ("Pattern not found", 1);
        }            

        if (sscanf(ptr, "%zu", &(rc->pattern[read++])) < 1) {
            error ("Failed to parse first pattern element", 1);
        }

        while ((ptr = strtok(NULL, delim)) && read < MAX_PATTERN_LEN) {
            if (sscanf(ptr, "%zu", &(rc->pattern[read++])) < 1) {
                error ("Failed to parse pattern", 1);
            }
        }
        rc->pattern_len = read;
    }
}


void print_run_config(struct run_config rc){
    printf("Index: %zu ", rc.pattern_len);
    printf("[");
    for (size_t i = 0; i < rc.pattern_len; i++) {
        printf("%zu", rc.pattern[i]);
        if (i != rc.pattern_len-1) printf(" ");
    }
    printf("]\n");
    if (rc.deltas_len > 0) {
        printf("Deltas: %zu ", rc.deltas_len);
        printf("[");
        for (size_t i = 0; i < rc.deltas_len; i++) {
            printf("%zu", rc.deltas[i]);
            if (i != rc.deltas_len-1) printf(" ");
        }
        printf("]\n");
        printf("Deltas_ps: %zu ", rc.deltas_len);
        printf("[");
        for (size_t i = 0; i < rc.deltas_len; i++) {
            printf("%zu", rc.deltas_ps[i]);
            if (i != rc.deltas_len-1) printf(" ");
        }
        printf("] (%zu)\n", rc.delta);
    } else {
        printf("Delta: %zu\n", rc.delta);
    }
    printf("kern: %s\n", kernel_name);
    printf("genlen: %zu\n", rc.generic_len);
}
