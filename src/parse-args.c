#include <getopt.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "parse-args.h"
#include "backend-support-tests.h"
#include "sp_alloc.h"
#include "json.h"

#ifdef USE_CUDA 
#include "../src/cuda/cuda-backend.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_PAPI
#include "papi_helper.h"
#endif

#define VALIDATE    1005
#define CLPLATFORM  1010
#define CLDEVICE    1011
#define PAPI_ARG    1012

#define INTERACTIVE "INTERACTIVE"

extern char platform_string[STRING_SIZE];
extern char device_string[STRING_SIZE];
extern char kernel_file[STRING_SIZE];
extern char kernel_name[STRING_SIZE];

extern int cuda_dev;
extern int validate_flag;
extern int quiet_flag;
extern int aggregate_flag;
extern int compress_flag;

#ifdef USE_PAPI
extern int papi_nevents;
extern char papi_event_names[PAPI_MAX_COUNTERS][STRING_SIZE];
#endif

extern enum sg_backend backend;

// These should actually stay global
int verbose;
FILE *err_file;

void error(char *what, int code);
void safestrcopy(char *dest, char *src);
void parse_p(char*, struct run_config *);
ssize_t setincludes(size_t key, size_t* set, size_t set_len);

char short_options[] = "W:l:k:qv:R:p:d:f:b:z:m:yw:t:n:aqcs:";
void parse_backend(int argc, char **argv);

char jsonfilename[STRING_SIZE];

int get_num_configs(json_value* value) {
    if (value->type != json_array) {
        error ("get_num_configs was not passed an array", ERROR);
    }

    return value->u.array.length;

}

struct run_config parse_json_config(json_value *value){
    struct run_config rc = {0};

    if (!value) {
        error ("parse_json_config passed NULL pointer", ERROR);
    }

    if (value->type != json_object) {
        error ("parse_json_config should only be passed json_objects", ERROR);
    }
    
    char **argv;
    int argc;
    argc = value->u.object.length + 1;
    argv = (char **)sp_malloc(sizeof(char*), argc*2, ALIGN_CACHE);
    for (int i = 0; i < argc; i++) {
        argv[i] = (char *)sp_malloc(1, STRING_SIZE*2, ALIGN_CACHE);
    }

    //json_value *values = value->u.object.values;

    for (int i = 0; i < argc-1; i++) {
        
        json_object_entry cur = value->u.object.values[i];

        if (cur.value->type == json_string) {
            snprintf(argv[i+1], STRING_SIZE, "--%s=%s", cur.name, cur.value->u.string.ptr);
        } else if (cur.value->type == json_integer) {
            snprintf(argv[i+1], STRING_SIZE, "--%s=%zd", cur.name, cur.value->u.integer);
        } else if (cur.value->type == json_array) {
           int index = 0;
           index += snprintf(argv[i+1], STRING_SIZE, "--%s=", cur.name);
           for (int j = 0; j < cur.value->u.array.length; j++) {
               if (cur.value->u.array.values[j]->type != json_integer) {
                   error ("encountered non-integer json type while parsing array", ERROR);
               }
               index += snprintf(&argv[i+1][index], STRING_SIZE-index, "%zd", cur.value->u.array.values[j]->u.integer);
               if (j != cur.value->u.array.length-1) {
                   index += snprintf(&argv[i+1][index], STRING_SIZE-index, ",");
               }

           }
        } else {
            error ("Unexpected json type", ERROR);
        }
    }
    //yeah its hacky - parse_args ignores the first arg
    safestrcopy(argv[0], argv[1]);

    rc = parse_runs(argc, argv);
    
    for (int i = 0; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);

    return rc;
}

void parse_args(int argc, char **argv, int *nrc, struct run_config **rc) 
{
    parse_backend(argc, argv); 

    int multi = 0;
    for (int i = 0; i < argc; i++) {
        if (strstr(argv[i], "-pFILE")) {
            safestrcopy(jsonfilename, strchr(argv[i],'=')+1);
            multi = 1;
            break;
        } else if (strstr(argv[i], "-p") &&  i < argc-1 && strstr(argv[i+1], "FILE")) {
            safestrcopy(jsonfilename, strchr(argv[i+1],'=')+1);
            multi = 1;
            break;
        }
    }

    if (multi) {
        FILE *fp; //j fopen(jsonfilename, "r");
        struct stat filestatus;
        int file_size;
        char *file_contents;
        json_char *json;
        json_value *value;

        if (stat(jsonfilename, &filestatus) != 0) {
            error ("Json file not found", ERROR);
        }

        file_size = filestatus.st_size;
        file_contents = (char *)sp_malloc(file_size, 1+1, ALIGN_CACHE);
        fp = fopen(jsonfilename, "rt");
        if (!fp) 
            error ("Unable to open Json file", ERROR);
        if (fread(file_contents, file_size, 1, fp) != 1) {
            fclose(fp);
            error ("Unable to read content of Json file", ERROR);
        }
        fclose(fp);

        json = (json_char*)file_contents;
        value = json_parse(json, file_size);

        if (!value) {
            error ("Unable to parse Json file", ERROR);
        }

        *nrc = get_num_configs(value);

        *rc = (struct run_config*)sp_calloc(sizeof(struct run_config), *nrc, ALIGN_CACHE);
        
        for (int i = 0; i < *nrc; i++) {
            rc[0][i] = parse_json_config(value->u.array.values[i]);
        }

        //exit(0);
        return;
    }
    *rc = (struct run_config*)sp_calloc(sizeof(struct run_config), 1, ALIGN_CACHE);
    rc[0][0] = parse_runs(argc, argv);
    *nrc = 1;
}

struct run_config parse_runs(int argc, char **argv)
{
    int supress_errors = 0;

    int pattern_found = 0;

    volatile char *argv0copy = argv[0];

    struct run_config rc = {0};
    rc.delta = -1;
#ifdef USE_OPENMP
    rc.omp_threads = omp_get_max_threads();
#else 
    rc.omp_threads = 1;
#endif
    rc.kernel = INVALID_KERNEL;
    safestrcopy(rc.name,"NONE");


    //Do NOT remove this - as we call getopt_long_only in multiple places, this
    //must be rest between calls. 
    optind = 0;
	static struct option long_options[] =
    {
        // Run Config 
        {"kernel-name",     required_argument, NULL, 'k'},
        {"pattern",         required_argument, NULL, 'p'},
        {"delta",           required_argument, NULL, 'd'},
        {"count",           required_argument, NULL, 'l'},
        {"wrap",            required_argument, NULL, 'w'},
        {"random",          required_argument, NULL, 's'},
        {"vector-len",      required_argument, NULL, 'v'},
        {"runs",            required_argument, NULL, 'R'},
        {"omp-threads",     required_argument, NULL, 't'},
        {"op",              required_argument, NULL, 'o'},
        {"local-work-size", required_argument, NULL, 'z'},
        {"shared-mem",      required_argument, NULL, 'm'},
        {"name",            required_argument, NULL, 'n'},
        {"papi",            required_argument, NULL, 0},
        {"cl-device",       required_argument, NULL, 0},
        {"verbose",         no_argument,       NULL, 0},
        {"aggregate",       optional_argument, NULL, 1},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, short_options,
                         long_options, &option_index);

        switch(c){
            case CLPLATFORM:
                safestrcopy(platform_string, optarg);
                break;
            case CLDEVICE:
                safestrcopy(device_string, optarg);
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
            case 'n':
                safestrcopy(rc.name, optarg);
                break;
            case 'p':
                safestrcopy(rc.generator, optarg);
                parse_p(optarg, &rc);
                pattern_found = 1;
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

    // VALIDATE ARGUMENTS

    if (!pattern_found) {
        error ("Please specify a pattern", ERROR);
    }
    if (rc.vector_len == 0) {
        error ("Vector length not set. Default is 1", WARN);
        rc.vector_len = 1;
    }

    if (rc.wrap == 0) {
        error ("length of smallbuf not specified. Default is 1 (slot of size pattern_len elements)", 0);
        rc.wrap = 1;
    }

    if (rc.nruns == 0) {
        error ("Number of runs not specified. Default is 10 ", 0);
        rc.nruns = 10;
    }

    if (rc.generic_len == 0) {
        error ("Length not specified. Default is 32 (gathers/scatters)", 0);
        rc.generic_len = 32;
    }


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

    if (rc.delta <= -1) {
        //printf("rc.delta: %zd\n");
        error("delta not specified, default is 8\n", WARN);
        rc.delta = 8;
        rc.deltas_len = 1;
    }
    
    if (rc.op != OP_COPY) {
        error("OP must be OP_COPY", WARN);
    }

    if (!strcasecmp(rc.name, "NONE")) {
        if (rc.type != CUSTOM) {
            safestrcopy(rc.name, rc.generator);
        } else {
            safestrcopy(rc.name, "CUSTOM");
        }
    }



#ifdef USE_OPENMP
    int max_threads = omp_get_max_threads();
    if (rc.omp_threads > max_threads) {
        error ("Too many OpenMP threads requested, using the max instead", WARN);
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

#if defined USE_CUDA || defined USE_OPENCL
    if (rc.local_work_size == 0) {
        error ("Local_work_size not set. Default is 1", WARN);
        rc.local_work_size = 1;
    }
#endif

    return rc;

}

void parse_backend(int argc, char **argv)
{
    err_file   = stderr;

    safestrcopy(platform_string, "NONE");
    safestrcopy(device_string,   "NONE");
    safestrcopy(kernel_file,     "NONE");
    safestrcopy(kernel_name,     "NONE");

    int supress_errors = 0;

    //Do NOT remove this - as we call getopt_long_only in multiple places, this
    //must be rest between calls. 
    optind = 1;
	static struct option long_options[] =
    {
        /* Output */
        {"no-print-header", no_argument,       NULL, 'q'},
        {"verbose",         no_argument,       &verbose, 1},
        /* Backend */
        {"backend",         required_argument, NULL, 'b'},
        {"cl-platform",     required_argument, NULL, CLPLATFORM},
        {"cl-device",       required_argument, NULL, CLDEVICE},
        {"kernel-file",     required_argument, NULL, 'f'},
        {"interactive",     no_argument,       NULL, 'i'},
        /* Other */
        {"validate",        no_argument, &validate_flag, 1},
        {"aggregate",       optional_argument, NULL, 'a'},
        {"compress",        optional_argument, NULL, 'c'},
        {"papi",            required_argument, NULL, PAPI_ARG},
        {"local-work-size", required_argument, NULL, 'z'},
        {0, 0, 0, 0}
    };  

    int c = 0;
    int option_index = 0;

    while(c != -1){

    	c = getopt_long_only (argc, argv, short_options,
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
            case 'q':
                quiet_flag ++;
                break;
            case 'a':
                if (optarg == NULL) {
                    aggregate_flag = 0;
                }else {
                    sscanf(optarg, "%d", &aggregate_flag);
                }
                break;
            case 'c':
                if (optarg == NULL) {
                    compress_flag = 1;
                }else {
                    sscanf(optarg, "%d", &compress_flag);
                }
                break;
            case PAPI_ARG:
                {
#ifdef USE_PAPI
                    char *pch = strtok(optarg, ",");
                    while (pch != NULL) {
                        safestrcopy(papi_event_names[papi_nevents++], pch);
                        pch = strtok (NULL, ",");
                        if(papi_nevents == PAPI_MAX_COUNTERS) break;
                    }
#endif
                }
                break;
            default:
                break;

        }

    }

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
        cuda_dev = dev;
        cudaSetDevice(dev);
    }
    #endif


    if (!strcasecmp(kernel_file, "NONE") && backend == OPENCL) {
        error("Kernel file unspecified, guessing kernels/kernels_vector.cl", 0);
        safestrcopy(kernel_file, "kernels/kernels_vector.cl");
    }


    return;
}


void parse_p(char* optarg, struct run_config *rc) {

    rc->type = INVALID_IDX;
    char *arg = 0;
    if ((arg=strchr(optarg, ':'))) {

        *arg = '\0';
        arg++; //arg now points to arguments to the pattern type

        // FILE mode indicates that we will load a 
        // config from a file
        if (!strcmp(optarg, "FILE")) {
            //TODO
            //safestrcopy(idx_pattern_file, arg);
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
            
            char *delta = strtok(NULL, ":");
            if (delta) {
                if (!strcmp(delta, "NR")) {
                    rc->delta = strideval*rc->pattern_len;
                    rc->deltas[0] = rc->delta;
                    rc->deltas_len = 1;
                } else {
                    if (sscanf(delta, "%zd", &(rc->delta)) < 1) {
                        error("UNIFORM: delta not parsed", 1);
                    }
                    rc->deltas[0] = rc->delta;
                    rc->deltas_len = 1;
                            
                }
            }
            

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
            /*
            printf("here\n");
            printf("gaps: %x\n", gaps);
            if (gaps == 0) {
                printf("1\n");
            } 
            if (!gaps) {
                printf("2\n");
                error("FUCK", 1);
                exit(1);
            }
            if (!gaps || (gaps && gaps[0] == '\0')) {
                error("MS1: Gaps missing", ERROR);
            }
            printf("here\n");

            */
            if(!gaps) {
                printf("1\n");
                exit(1);
                error("error", ERROR);
            }

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

    if (rc->pattern_len == 0) {
        error("Pattern length of 0", ERROR);
    }
    if (rc->type == INVALID_IDX) {
        error("No pattern type set", ERROR);
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
    if (code == ERROR) {
        fprintf(err_file, "Error: ");
    }
    else if (code == WARN){
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
