#include <getopt.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "parse-args.h"
#include "backend-support-tests.h"

#ifdef USE_CUDA 
#include "cuda-backend.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define VALIDATE    1005
#define CLPLATFORM  1010
#define CLDEVICE    1011

#define INTERACTIVE "INTERACTIVE"

extern char platform_string[STRING_SIZE];
extern char device_string[STRING_SIZE];
extern char kernel_file[STRING_SIZE];
extern char kernel_name[STRING_SIZE];

extern size_t seed;
extern int validate_flag;
extern int print_header_flag;

extern enum sg_backend backend;

// These should actually stay global
int verbose;
FILE *err_file;

void safestrcopy(char *dest, char *src);
void parse_p(char*, struct run_config *);
ssize_t setincludes(size_t key, size_t* set, size_t set_len);

struct run_config parse_args(int argc, char **argv)
{
    struct backend_config bc;

    bc.backend = INVALID_BACKEND;
    
    err_file   = stderr;

    safestrcopy(platform_string, "NONE");
    safestrcopy(device_string,   "NONE");
    safestrcopy(kernel_file,     "NONE");
    safestrcopy(kernel_name,     "NONE");

    int supress_errors = 0;

    struct run_config rc = {0};
    rc.delta = -1;
    rc.kernel = INVALID_KERNEL;

	static struct option long_options[] =
    {
        /* Output */
        {"no-print-header", no_argument, &print_header_flag, 0},
        {"nph",             no_argument, &print_header_flag, 0},
        {"supress-errors",  no_argument,       NULL, 'q'},
        {"verbose",         no_argument,       &verbose, 1},
        /* Backend */
        {"backend",         required_argument, NULL, 'b'},
        {"cl-platform",     required_argument, NULL, CLPLATFORM},
        {"cl-device",       required_argument, NULL, CLDEVICE},
        {"kernel-file",     required_argument, NULL, 'f'},
        {"kernel-name",     required_argument, NULL, 'k'},
        {"interactive",     no_argument,       NULL, 'i'},
        /* Run Config */
        {"pattern",         required_argument, NULL, 'p'},
        {"delta",           required_argument, NULL, 'd'},
        {"generic-len",     required_argument, NULL, 'l'},
        {"wrap",            required_argument, NULL, 'w'},
        {"random",          required_argument, NULL, 's'},
        {"vector-len",      required_argument, NULL, 'v'},
        {"runs",            required_argument, NULL, 'R'},
        {"omp-threads",     required_argument, NULL, 't'},
        {"op",              required_argument, NULL, 'o'},
        {"local-work-size", required_argument, NULL, 'z'},
        {"shared-mem",      required_argument, NULL, 'm'},
        /* Other */
        {"validate",        no_argument, &validate_flag, 1},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, "W:l:k:qv:R:p:d:f:b:z:m:yw:t:",
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
                else if(!strcasecmp("SERIAL", optarg)){
                    backend = SERIAL;
                }
                else {
                    error ("Unrecognized Backend", ERROR);
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
                    rc.kernel=SG;
                }
                else if (!strcasecmp("SCATTER", optarg)) {
                    rc.kernel=SCATTER;
                }
                else if (!strcasecmp("GATHER", optarg)) {
                    rc.kernel=GATHER;
                }
                else {
                    error("Invalid kernel", 1);
                }
                break;
            // run config
            case 'o':
                if (!strcasecmp("COPY", optarg)) { 
                    rc.op = OP_COPY;
                } else if (!strcasecmp("ACCUM", optarg)) {
                    rc.op = OP_ACCUM;
                } else {
                    error("Unrecognzied op type", ERROR);
                }
                break;
            case 's':
                sscanf(optarg, "%zu", &rc.random_seed);
                break;
            case 't':
                sscanf(optarg, "%zu", &rc.omp_threads);
                break;
            case 'v':
                sscanf(optarg, "%zu", &rc.vector_len);
                if (rc.vector_len < 1) {
                    error("Invalid vector len", 1);
                }
                break;
            case 'R':
                sscanf(optarg, "%zu", &rc.nruns);
                break;
            case 'w':
                sscanf(optarg,"%zu", &rc.wrap);
                break;
            case 'l':
                sscanf(optarg,"%zu", &(rc.generic_len));
                break;
            case 'z':
                sscanf(optarg,"%zu", &rc.local_work_size);
                break;
            case 'm':
                sscanf(optarg,"%u", &rc.shmem);
                break;
            case 'q':
                err_file = fopen("/dev/null", "w");
                break;
            case 'p':
                parse_p(optarg, &rc);
                break;
            case 'd':
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
                    //printf("rc.deltas_ps[%zu] = %zu\n",i, rc.deltas_ps[i]);
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
            default:
                break;

        }

    }

    if (rc.wrap == 0) {
        error ("length of smallbuf not specified. Default is 1 (slot of size pattern_len elements)", 0);
        rc.wrap = 1;
    }

    if (rc.nruns == 0) {
        error ("Number of runs not specified. Default is 10 ", 0);
        rc.nruns = 10;
    }

    if (rc.generic_len <= 0) {
        error ("Length not specified. Default is 32 (elements)", 0);
        rc.generic_len = 32;
    }

#ifdef USE_OPENMP
    int max_threads = omp_get_max_threads();
    if (rc.omp_threads > max_threads) {
        error ("Too many OpenMP threads requested, using the max instead", ERROR);
    }
    if (rc.omp_threads == 0) {
        error ("Number of OpenMP threads not specified, using the max", WARN);
        rc.omp_threads = max_threads;
    }
#else
    if (rc.omp_threads > 1) {
        error ("Compiled without OpenMP support but requsted more than 1 thread, using 1 instead", WARN);
    }
#endif


    /* Check argument coherency */
    if(backend == INVALID_BACKEND){
        if (sg_cuda_support()) {
            backend = CUDA;
            error ("No backend specified, guessing CUDA", WARN);
        }
        else if (sg_opencl_support()) {
            backend = OPENCL;
            error ("No backend specified, guessing OpenCL", WARN);
        }
        else if (sg_openmp_support()) { 
            backend = OPENMP;
            error ("No backend specified, guessing OpenMP", WARN);
        }
        else if (sg_serial_support()) { 
            backend = SERIAL;
            error ("No backend specified, guessing Serial", WARN);
        }
        else
        {
            error ("No backends available! Please recompile spatter with at least one backend.", ERROR);
        }
    }

    // Check to see if they compiled with support for their requested backend
    if(backend == OPENCL){
        if (!sg_opencl_support()) {
            error("You did not compile with support for OpenCL", ERROR);
        }
    }
    else if(backend == OPENMP){
        if (!sg_openmp_support()) {
            error("You did not compile with support for OpenMP", ERROR);
        }
    }
    else if(backend == CUDA){
        if (!sg_cuda_support()) {
            error("You did not compile with support for CUDA", ERROR);
        }
    }
    else if(backend == SERIAL){
        if (!sg_serial_support()) {
            error("You did not compile with support for serial execution", ERROR);
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

    if (rc.kernel == INVALID_KERNEL) {
        error("Kernel unspecified, guess GATHER", WARN);
        rc.kernel = GATHER;
        safestrcopy(kernel_name, "gather");
    }

    if (rc.kernel == SCATTER) {
        sprintf(kernel_name, "%s%zu", "scatter", rc.vector_len);
    } else if (rc.kernel == GATHER) {
        sprintf(kernel_name, "%s%zu", "gather", rc.vector_len);
    } else if (rc.kernel == SG) {
        sprintf(kernel_name, "%s%zu", "sg", rc.vector_len);
    }

    if (!strcasecmp(kernel_file, "NONE") && backend == OPENCL) {
        error("Kernel file unspecified, guessing kernels/kernels_vector.cl", 0);
        safestrcopy(kernel_file, "kernels/kernels_vector.cl");
    }

    if (rc.delta == -1) {
        error("delta not specified, default is 8\n", 0);
        rc.delta = 8;
    }

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

ssize_t setincludes(size_t key, size_t* set, size_t set_len){ 
    for (size_t i = 0; i < set_len; i++) {
        if (set[i] == key) {
            return i;
        }
    }
    return -1;
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
