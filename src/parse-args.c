#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "parse-args.h"
#include "backend-support-tests.h"
#include "sp_alloc.h"
#include "json.h"
#include "pcg_basic.h"
#include "argtable3.h"

#ifdef USE_CUDA
#include "../src/cuda/cuda-backend.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_PAPI
#include "papi_helper.h"
int papi_nevents;
char papi_event_names[PAPI_MAX_COUNTERS][STRING_SIZE];
#endif

#define INTERACTIVE "INTERACTIVE"

char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];
char jsonfilename[STRING_SIZE];
char op_string[STRING_SIZE];

int cuda_dev = -1;
int validate_flag = 0;
int quiet_flag = 0;
int aggregate_flag = 1;
int compress_flag = 0;
int stride_kernel = -1;

enum sg_backend backend = INVALID_BACKEND;

// These should actually stay global
int verbose;
FILE *err_file;

void safestrcopy(char *dest, const char *src);
void parse_p(char*, struct run_config *);
ssize_t setincludes(size_t key, size_t* set, size_t set_len);
void xkp_pattern(size_t *pat, size_t dim);
void parse_backend(int argc, char **argv);

void** argtable;
unsigned int number_of_arguments = 31;
struct arg_lit *verb, *help, *interactive, *validate, *aggregate, *compress;
struct arg_str *backend_arg, *cl_platform, *cl_device, *pattern, *kernelName, *delta, *name, *papi, *op;
struct arg_int *count, *wrap, *runs, *omp_threads, *vector_len, *local_work_size, *shared_memory, *morton, *hilbert, *roblock, *stride, *random_arg, *no_print_header;
struct arg_file *kernelFile;
struct arg_end *end;

void initialize_argtable()
{
    // Initialize the argtable on the stack just because it is easier and how the documentation handles it
    void** malloc_argtable = (void**) malloc(sizeof(void*) * number_of_arguments);

    // Arguments that do not take parameters
    malloc_argtable[0] = help            = arg_litn(NULL, "help", 0, 1, "Displays info about commands and then exits.");
    malloc_argtable[1] = verb            = arg_litn(NULL, "verbose", 0, 1, "Print info about default arguments that you have not overridden.");
    malloc_argtable[2] = no_print_header = arg_intn("q", "no-print-header", "<n>", 0, 1, "Do not print header information.");
    malloc_argtable[3] = interactive     = arg_litn("i", "interactive", 0, 1, "Pick the platform and the device interactively.");
    malloc_argtable[4] = validate        = arg_litn(NULL, "validate", 0, 1, "Perform extra validation checks to ensure data validity");
    malloc_argtable[5] = aggregate       = arg_litn("a", "aggregate", 0, 1, "Report a minimum time for all runs of a given configuration for 2 or more runs. [Default 1] (Do not use with PAPI)");
    malloc_argtable[6] = compress        = arg_litn("c", "compress", 0, 1, "TODO");
    // Benchmark Configuration
    malloc_argtable[7] = pattern         = arg_strn("p", "pattern", "<pattern>", 1, 1, "Specify either a a built-in pattern (i.e. UNIFORM), a custom pattern (i.e. 1,2,3,4), or a path to a json file with a run-configuration.");
    malloc_argtable[8] = kernelName      = arg_strn("k", "kernel-name", "<kernel>", 0, 1, "Specify the kernel you want to run. [Default: Gather]");
    malloc_argtable[9] = op              = arg_strn("o", "op", "<s>", 0, 1, "TODO");
    malloc_argtable[10] = delta           = arg_strn("d", "delta", "<delta[,delta,...]>", 0, 1, "Specify one or more deltas. [Default: 8]");
    malloc_argtable[11] = count           = arg_intn("l", "count", "<n>", 0, 1, "Number of Gathers or Scatters to perform.");
    malloc_argtable[12] = wrap            = arg_intn("w", "wrap", "<n>", 0, 1, "Number of independent slots in the small buffer (source buffer if Scatter, Target buffer if Gather. [Default: 1]");
    malloc_argtable[13] = runs            = arg_intn("R", "runs", "<n>", 0, 1, "Number of times to repeat execution of the kernel. [Default: 10]");
    malloc_argtable[14] = omp_threads     = arg_intn("t", "omp-threads", "<n>", 0, 1, "Number of OpenMP threads. [Default: OMP_MAX_THREADS]");
    malloc_argtable[15] = vector_len      = arg_intn("v", "vector-len", "<n>", 0, 1, "TODO");
    malloc_argtable[16] = local_work_size = arg_intn("z", "local-work-size", "<n>", 0, 1, "Numer of Gathers or Scatters performed by each thread on a GPU.");
    malloc_argtable[17] = shared_memory   = arg_intn("m", "shared-memory", "<n>", 0, 1, "Amount of dummy shared memory to allocate on GPUs (used for occupancy control).");
    malloc_argtable[18] = name            = arg_strn("n", "name", "<name>", 0, 1, "Specify and name this configuration in the output.");
    malloc_argtable[19] = random_arg      = arg_intn("s", "random", "<n>", 0, 1, "Sets the seed, or uses a random one if no seed is specified.");
    // Backend Configuration
    malloc_argtable[20] = backend_arg     = arg_strn("b", "backend", "<backend>", 0, 1, "Specify a backend: OpenCL, OpenMP, CUDA, or Serial.");
    malloc_argtable[21] = cl_platform     = arg_strn(NULL, "cl-platform", "<platform>", 0, 1, "Specify platform if using OpenCL (case-insensitive, fuzzy matching).");
    malloc_argtable[22] = cl_device       = arg_strn(NULL, "cl-device", "<device>", 0, 1, "Specify device if using OpenCL (case-insensitive, fuzzy matching).");
    malloc_argtable[23] = kernelFile      = arg_filen("f", "kernel-file", "<FILE>", 0, 1, "Specify the location of an OpenCL kernel file.");
    // Other Configurations
    malloc_argtable[24] = morton          = arg_intn(NULL, "morton", "<n>", 0, 1, "TODO");
    malloc_argtable[25] = hilbert         = arg_intn(NULL, "hilbert", "<n>", 0, 1, "TODO");
    malloc_argtable[26] = roblock         = arg_intn(NULL, "roblock", "<n>", 0, 1, "TODO");
    malloc_argtable[27] = stride          = arg_intn(NULL, "stride", "<n>", 0, 1, "TODO");
    malloc_argtable[28] = papi            = arg_strn(NULL, "papi", "<s>", 0, 1, "TODO");
    malloc_argtable[29] = end             = arg_end(20);

    // Random has an option to provide an argument. Default its value to -1.
    random_arg->hdr.flag |= ARG_HASOPTVALUE;
    random_arg->ival[0] = -1;

    // Set default values
    kernelName->sval[0] = "Gather\0";
    delta->sval[0] = "8\0";
    wrap->ival[0] = 1;
    runs->ival[0] = 10;

    // Set the global argtable equal to the malloc argtable
    argtable = malloc_argtable;
}


void copy_str_ignore_leading_space(char* dest, const char* source)
{
    if (source[0] == ' ')
        safestrcopy(dest, &source[1]);
    else
        safestrcopy(dest, source);
}

int get_num_configs(json_value* value)
{
    if (value->type != json_array) {
        error("get_num_configs was not passed an array", ERROR);
    }

    return value->u.array.length;
}

void parse_json_kernel(json_object_entry cur, char** argv, int i)
{
    if (!strcasecmp(cur.value->u.string.ptr, "SCATTER") || !strcasecmp(cur.value->u.string.ptr, "GATHER") || !strcasecmp(cur.value->u.string.ptr, "SP"))
    {
        error("Ambiguous Kernel Type: Assuming kernel-name option.", WARN);
        snprintf(argv[i+1], STRING_SIZE, "--kernel-name=%s", cur.value->u.string.ptr);
    }
    else
    {
        error("Ambigous Kernel Type: Assuming kernel-file option.", WARN);
        snprintf(argv[i+1], STRING_SIZE, "--kernel-file=%s", cur.value->u.string.ptr);
    }
}

void parse_json_array(json_object_entry cur, char** argv, int i)
{
    int index = 0;
    index += snprintf(argv[i+1], STRING_SIZE, "--%s=", cur.name);

    for (int j = 0; j < cur.value->u.array.length; j++)
    {
        if (cur.value->u.array.values[j]->type != json_integer)
        {
            error ("Encountered non-integer json type while parsing array", ERROR);
        }
	char buffer[50];
	int check = snprintf(buffer, 50, "%zd", cur.value->u.array.values[j]->u.integer);
        int added = snprintf(buffer, STRING_SIZE-index, "%zd", cur.value->u.array.values[j]->u.integer);
	
	if (check == added) {
	    index += snprintf(&argv[i+1][index], STRING_SIZE-index, "%zd", cur.value->u.array.values[j]->u.integer);
	    
	    if (index >= STRING_SIZE-1) break;
	    else if (j != cur.value->u.array.length-1 && index < STRING_SIZE-1) {
                index += snprintf(&argv[i+1][index], STRING_SIZE-index, ",");
            }
        }
	else {
		index--;
		argv[i+1][index] = '\0';
		break;
	}

   }
}

struct run_config parse_json_config(json_value *value)
{
    struct run_config rc = {0};

    if (!value)
        error ("parse_json_config passed NULL pointer", ERROR);

    if (value->type != json_object)
        error ("parse_json_config should only be passed json_objects", ERROR);

    int argc = value->u.object.length + 1;
    char **argv = (char **)sp_malloc(sizeof(char*), argc*2, ALIGN_CACHE);

    for (int i = 0; i < argc; i++)
        argv[i] = (char *)sp_malloc(1, STRING_SIZE*2, ALIGN_CACHE);

    for (int i = 0; i < argc-1; i++)
    {
        json_object_entry cur = value->u.object.values[i];

        if (cur.value->type == json_string)
        {
            if (!strcasecmp(cur.name, "kernel"))
            {
                parse_json_kernel(cur, argv, i);
            }
            else
            {
                snprintf(argv[i+1], STRING_SIZE, "--%s=%s", cur.name, cur.value->u.string.ptr);
            }
        }
        else if (cur.value->type == json_integer)
        {
            snprintf(argv[i+1], STRING_SIZE, "--%s=%zd", cur.name, cur.value->u.integer);
        }
        else if (cur.value->type == json_array)
        {
            parse_json_array(cur, argv, i);
        }
        else
        {
            error ("Unexpected json type", ERROR);
        }
    }

    //yeah its hacky - parse_args ignores the first arg
    safestrcopy(argv[0], argv[1]);

    int nerrors = arg_parse(argc, argv, argtable);

    if (nerrors > 0)
    {
        arg_print_errors(stdout, end, "Spatter");
        printf("Error while parsing json file.\n");
        exit(0);
    }

    rc = parse_runs(argc, argv);

    for (int i = 0; i < argc; i++)
        free(argv[i]);

    free(argv);

    return rc;
}

void parse_args(int argc, char **argv, int *nrc, struct run_config **rc)
{
    initialize_argtable();
    int nerrors = arg_parse(argc, argv, argtable);

    if (help->count > 0)
    {
        printf("Usage:\n");
        arg_print_syntax(stdout, argtable, "\n");
        arg_print_glossary(stdout, argtable, " %-28s %s\n");
        exit(0);
    }

    if (nerrors > 0)
    {
        arg_print_errors(stdout, end, "Spatter");
        printf("Try './spatter --help' for more information.\n");
        exit(0);
    }

    parse_backend(argc, argv);


    // Parse command-line arguments to in case of specified json file.
    int json = 0;

   if (pattern->count > 0)
   {
       if (strstr(pattern->sval[0], "FILE"))
       {
           safestrcopy(jsonfilename, strchr(pattern->sval[0], '=') + 1);
           printf("Reading patterns from %s.\n", jsonfilename);
           json = 1;
       }
   }

    if (json)
    {
        FILE *fp;
        struct stat filestatus;
        int file_size;
        char *file_contents;
        json_char *json;
        json_value *value;

        if (stat(jsonfilename, &filestatus) != 0)
            error ("Json file not found", ERROR);

        file_size = filestatus.st_size;
        file_contents = (char *)sp_malloc(file_size, 1+1, ALIGN_CACHE);

        fp = fopen(jsonfilename, "rt");
        if (!fp)
            error ("Unable to open Json file", ERROR);

        if (fread(file_contents, file_size, 1, fp) != 1)
        {
            fclose(fp);
            error ("Unable to read content of Json file", ERROR);
        }
        fclose(fp);

        json = (json_char*)file_contents;
        value = json_parse(json, file_size);

        if (!value)
            error ("Unable to parse Json file", ERROR);

        // This is the number of specified runs in the json file.
        *nrc = get_num_configs(value);

        *rc = (struct run_config*)sp_calloc(sizeof(struct run_config), *nrc, ALIGN_CACHE);

        for (int i = 0; i < *nrc; i++)
            rc[0][i] = parse_json_config(value->u.array.values[i]);

        json_value_free(value);
        free(file_contents);
    }
    else
    {
        *rc = (struct run_config*)sp_calloc(sizeof(struct run_config), 1, ALIGN_CACHE);
        rc[0][0] = parse_runs(argc, argv);
        *nrc = 1;
    }

    free(argtable);

    return;
}

struct run_config parse_runs(int argc, char **argv)
{
    int pattern_found = 0;

    struct run_config rc = {0};
    rc.delta = -1;
    rc.stride_kernel = -1;
    rc.ro_block = 1;
    rc.ro_order = NULL;
#ifdef USE_OPENMP
    rc.omp_threads = omp_get_max_threads();
#else
    rc.omp_threads = 1;
#endif
    rc.kernel = INVALID_KERNEL;
    safestrcopy(rc.name,"NONE");

   if (kernelName->count > 0)
   {
        copy_str_ignore_leading_space(kernel_name, kernelName->sval[0]);
        if (!strcasecmp("SG", kernel_name))
            rc.kernel=SG;
        else if (!strcasecmp("SCATTER", kernel_name))
            rc.kernel=SCATTER;
        else if (!strcasecmp("GATHER", kernel_name))
            rc.kernel=GATHER;
        else
        {
            char output[STRING_SIZE];
            sprintf(output, "Invalid kernel %s\n", kernel_name);
            error(output, ERROR);
        }
   }

   if (op->count > 0)
   {
        copy_str_ignore_leading_space(op_string, op->sval[0]);
        if (!strcasecmp("COPY", op_string))
            rc.op = OP_COPY;
        else if (!strcasecmp("ACCUM", op_string))
            rc.op = OP_ACCUM;
        else
            error("Unrecognzied op type", ERROR);
   }

   if (random_arg->count > 0)
   {
        // Parsing the seed parameter
        // If no argument was passed, use the current time in seconds since the epoch as the random seed
        if (random_arg->ival[0] == -1)
            rc.random_seed = time(NULL);
        else
            //sscanf(optarg, "%zu", &rc.random_seed);
            rc.random_seed = random_arg->ival[0];
   }

    if (omp_threads->count > 0)
        rc.omp_threads = omp_threads->ival[0];

    if (vector_len->count > 0)
    {
        rc.vector_len = vector_len->ival[0];
        if (rc.vector_len < 1)
            error("Invalid vector len!", ERROR);
    }

    if (runs->count > 0)
        rc.nruns = runs->ival[0];

    if (wrap->count > 0)
        rc.wrap = wrap->ival[0];

    if (count->count > 0)
        rc.generic_len = count->ival[0];

    if (local_work_size->count > 0)
        rc.local_work_size = local_work_size->ival[0];

    if (shared_memory->count > 0)
        rc.shmem = shared_memory->ival[0];

    if (name->count > 0)
        copy_str_ignore_leading_space(rc.name, name->sval[0]);

    if (pattern->count > 0)
    {
        copy_str_ignore_leading_space(rc.generator, pattern->sval[0]);
        //char* filePtr = strstr(rc.generator, "FILE");
        //if (filePtr)
        //    safestrcopy(rc.generator, filePtr);
        parse_p(rc.generator, &rc);
        pattern_found = 1;
    }

    if (delta->count > 0)
    {
        char delta_temp[STRING_SIZE];
        copy_str_ignore_leading_space(delta_temp, delta->sval[0]);
        char *delim = ",";
        char *ptr = strtok(delta_temp, delim);
        size_t read = 0;
        if (!ptr)
            error("Pattern not found", ERROR);

        if (sscanf(ptr, "%zu", &(rc.deltas[read++])) < 1)
            error("Failed to parse first pattern element in deltas", ERROR);

        spIdx_t *mydeltas;
        spIdx_t *mydeltas_ps;

        mydeltas = sp_malloc(sizeof(size_t), MAX_PATTERN_LEN, ALIGN_CACHE);
        mydeltas_ps = sp_malloc(sizeof(size_t), MAX_PATTERN_LEN, ALIGN_CACHE);

        while ((ptr = strtok(NULL, delim)) && read < MAX_PATTERN_LEN)
        {
            if (sscanf(ptr, "%zu", &(rc.deltas[read++])) < 1)
                error("Failed to parse pattern", ERROR);
        }
        rc.deltas = mydeltas;
        rc.deltas_ps = mydeltas_ps;
        rc.deltas_len = read;

        // rotate
        for (size_t i = 0; i < rc.deltas_len; i++)
            rc.deltas_ps[i] = rc.deltas[((i-1)+rc.deltas_len)%rc.deltas_len];

        // compute prefix-sum
        for (size_t i = 1; i < rc.deltas_len; i++)
            rc.deltas_ps[i] += rc.deltas_ps[i-1];

        // compute max
        size_t m = rc.deltas_ps[0];
        for (size_t i = 1; i < rc.deltas_len; i++)
        {
            if (rc.deltas_ps[i] > m)
                m = rc.deltas_ps[i];
        }
        rc.delta = m;
    }

    if (morton->count > 0)
        rc.ro_morton = morton->ival[0];

    if (hilbert->count > 0)
        rc.ro_hilbert = hilbert->ival[0];

    if (roblock->count > 0)
        rc.ro_block = roblock->ival[0];

    if (stride->count > 0)
        rc.stride_kernel = stride->ival[0];

    // VALIDATE ARGUMENTS
    if (!pattern_found)
        error ("Please specify a pattern", ERROR);

    if (rc.vector_len == 0)
    {
        error ("Vector length not set. Default is 1", WARN);
        rc.vector_len = 1;
    }

    if (rc.wrap == 0)
    {
        error ("length of smallbuf not specified. Default is 1 (slot of size pattern_len elements)", WARN);
        rc.wrap = 1;
    }

    if (rc.nruns == 0)
    {
        error ("Number of runs not specified. Default is 10 ", WARN);
        rc.nruns = 10;
    }

    if (rc.generic_len == 0)
    {
        error ("Length not specified. Default is 1024 (gathers/scatters)", WARN);
        rc.generic_len = 1024;
    }

    if (rc.kernel == INVALID_KERNEL)
    {
        error("Kernel unspecified, guess GATHER", WARN);
        rc.kernel = GATHER;
        safestrcopy(kernel_name, "gather");
    }

    if (rc.kernel == SCATTER)
        sprintf(kernel_name, "%s%zu", "scatter", rc.vector_len);
    else if (rc.kernel == GATHER)
        sprintf(kernel_name, "%s%zu", "gather", rc.vector_len);
    else if (rc.kernel == SG)
        sprintf(kernel_name, "%s%zu", "sg", rc.vector_len);

    if (rc.delta <= -1)
    {
        error("delta not specified, default is 8\n", WARN);
        rc.delta = 8;
        rc.deltas_len = 1;
    }

    if (rc.op != OP_COPY)
        error("OP must be OP_COPY", WARN);

    if (!strcasecmp(rc.name, "NONE"))
    {
        if (rc.type != CUSTOM)
            safestrcopy(rc.name, rc.generator);
        else
            safestrcopy(rc.name, "CUSTOM");
    }

#ifdef USE_OPENMP
    int max_threads = omp_get_max_threads();
    if (rc.omp_threads > max_threads)
    {
        error ("Too many OpenMP threads requested, using the max instead", WARN);
        rc.omp_threads = max_threads;
    }
    if (rc.omp_threads == 0)
    {
        error ("Number of OpenMP threads not specified, using the max", WARN);
        rc.omp_threads = max_threads;
    }
#else
    if (rc.omp_threads > 1)
        error ("Compiled without OpenMP support but requsted more than 1 thread, using 1 instead", WARN);
#endif

#if defined USE_CUDA || defined USE_OPENCL
    if (rc.local_work_size == 0)
    {
        error ("Local_work_size not set. Default is 1", WARN);
        rc.local_work_size = 1;
    }
#endif
    return rc;
}

ssize_t power(int base, int exp) {
    int i, result = 1;
    for (i = 0; i < exp; i++)
        result *= base;
    return result;
}

// Yes, there is no need for recursion here but I did this in python first. I'll
// updatte this later with a cleaner implementation
void static laplacian_branch(int depth, int order, int n, int **pos, int *pos_len)
{
    *pos = (int*)realloc(*pos, ((*pos_len)+order) * sizeof(int));

    for (int i = 0; i < order; i++) {
        (*pos)[i+*pos_len] = (i+1) * power(n, depth);
    }

    *pos_len += order;
    return;
}

void static laplacian(int dim, int order, int n, struct run_config *rc)
{

    if (dim < 1) {
        error("laplacian: dim must be positive", ERROR);
    }

    int final_len = dim * order * 2 + 1;
    if (final_len > MAX_PATTERN_LEN) {
        error("laplacian: resulting pattern too long", ERROR);
    }

    int pos_len = 0;
    int *pos = NULL;

    for (int i = 0; i < dim; i++) {
        laplacian_branch(i, order, n, &pos, &pos_len);
    }

    rc->pattern_len = final_len;
    int max = pos[pos_len-1];

    for (int i = 0; i < rc->pattern_len; i++) {
        rc->pattern[i] = 2;
    }

    //populate rc->pattern
    for(int i = 0; i < pos_len; i++) {
        rc->pattern[i] = (-pos[pos_len - i - 1] + max);
    }

    rc->pattern[pos_len] = max;

    for(int i = 0; i < pos_len; i++) {
        rc->pattern[pos_len+1+i] = pos[i] + max;
    }

    return;
}

void parse_backend(int argc, char **argv)
{
    err_file = stderr;

    safestrcopy(platform_string, "NONE");
    safestrcopy(device_string,   "NONE");
    safestrcopy(kernel_file,     "NONE");
    safestrcopy(kernel_name,     "NONE");

   if (backend_arg->count > 0)
   {
        if(!strcasecmp("OPENCL", backend_arg->sval[0]))
            backend = OPENCL;
        else if(!strcasecmp("OPENMP", backend_arg->sval[0]))
            backend = OPENMP;
        else if(!strcasecmp("CUDA", backend_arg->sval[0]))
            backend = CUDA;
        else if(!strcasecmp("SERIAL", backend_arg->sval[0]))
            backend = SERIAL;
        else
            error ("Unrecognized Backend", ERROR);
   }

   if (cl_platform->count > 0)
        copy_str_ignore_leading_space(platform_string, cl_platform->sval[0]);

    if (cl_device->count > 0)
        copy_str_ignore_leading_space(device_string, cl_device->sval[0]);

    if (interactive->count > 0)
    {
        safestrcopy(platform_string, INTERACTIVE);
        safestrcopy(device_string, INTERACTIVE);
    }

    if (kernelFile->count > 0)
        copy_str_ignore_leading_space(kernel_file, kernelFile->filename[0]);

    if (no_print_header->count > 0)
        quiet_flag = no_print_header->ival[0];

    if (validate->count > 0)
        validate_flag++;

    if (aggregate->count > 0)
        aggregate_flag = 1;

    if (compress->count > 0)
        compress_flag = 1;

    if (papi->count > 0)
    {
        #ifdef USE_PAPI
        {
            char *pch = strtok(papi->sval[0], ",");
            while (pch != NULL)
            {
                safestrcopy(papi_event_names[papi_nevents++], pch);
                pch = strtok (NULL, ",");
                if (papi_nevents == PAPI_MAX_COUNTERS)
                    break;
            }
        }
        #endif
    }

    /* Check argument coherency */
    if (backend == INVALID_BACKEND){
        if (sg_cuda_support())
        {
            backend = CUDA;
            error ("No backend specified, guessing CUDA", WARN);
        }
        else if (sg_opencl_support())
        {
            backend = OPENCL;
            error ("No backend specified, guessing OpenCL", WARN);
        }
        else if (sg_openmp_support())
        {
            backend = OPENMP;
            error ("No backend specified, guessing OpenMP", WARN);
        }
        else if (sg_serial_support())
        {
            backend = SERIAL;
            error ("No backend specified, guessing Serial", WARN);
        }
        else
            error ("No backends available! Please recompile spatter with at least one backend.", ERROR);
    }

    // Check to see if they compiled with support for their requested backend
    if (backend == OPENCL)
    {
        if (!sg_opencl_support())
            error("You did not compile with support for OpenCL", ERROR);
    }
    else if (backend == OPENMP)
    {
        if (!sg_openmp_support())
            error("You did not compile with support for OpenMP", ERROR);
    }
    else if (backend == CUDA)
    {
        if (!sg_cuda_support())
            error("You did not compile with support for CUDA", ERROR);
    }
    else if (backend == SERIAL)
    {
        if (!sg_serial_support())
            error("You did not compile with support for serial execution", ERROR);
    }

    if (backend == OPENCL)
    {
        if (!strcasecmp(platform_string, "NONE"))
        {
            safestrcopy(platform_string, INTERACTIVE);
            safestrcopy(device_string, INTERACTIVE);
        }
        if (!strcasecmp(device_string, "NONE"))
        {
            safestrcopy(platform_string, INTERACTIVE);
            safestrcopy(device_string, INTERACTIVE);
        }
    }

    #ifdef USE_CUDA
    if (backend == CUDA)
    {
        int dev = find_device_cuda(device_string);
        if (dev == -1)
        {
            error("Specified CUDA device not found or no device specified. Using device 0", WARN);
            dev = 0;
        }
        cuda_dev = dev;
        cudaSetDevice(dev);
    }
    #endif

    if (!strcasecmp(kernel_file, "NONE") && backend == OPENCL)
    {
        error("Kernel file unspecified, guessing kernels/kernels_vector.cl", WARN);
        safestrcopy(kernel_file, "kernels/kernels_vector.cl");
    }

    return;
}

void parse_p(char* optarg, struct run_config *rc)
{
    rc->type = INVALID_IDX;
    char *arg = 0;
    if ((arg=strchr(optarg, ':')))
    {
        *arg = '\0';
        arg++; //arg now points to arguments to the pattern type

        // FILE mode indicates that we will load a
        // config from a file
        if (!strcmp(optarg, "FILE"))
        {
            //TODO
            //safestrcopy(idx_pattern_file, arg);
            rc->type = CONFIG_FILE;
        }

        // The Exxon Kernel Proxy-derived stencil
        // It used to be called HYDRO so we will accept that too
        // XKP:dim
        else if (!strcmp(optarg, "XKP") || !strcmp(optarg, "HYDRO"))
        {
            rc->type = XKP;

            size_t dim = 0;
            char *dim_char = strtok(arg, ":");
            if (!dim_char)
                error("XKP: size not found", 1);
            if (sscanf(dim_char, "%zu", &dim) < 1)
                error("XKP: Dimension not parsed", 1);

            rc->pattern_len = 73;

            // The default delta is 1
            rc->delta = 1;
            rc->deltas[0] = rc->delta;
            rc->deltas_len = 1;

            xkp_pattern(rc->pattern, dim);
        }

        // Parse Uniform Stride Arguments, which are
        // UNIFORM:index_length:stride
        else if (!strcmp(optarg, "UNIFORM"))
        {
            rc->type = UNIFORM;

            // Read the length
            char *len = strtok(arg,":");
            if (!len)
                error("UNIFORM: Index Length not found", 1);
            if (sscanf(len, "%zu", &(rc->pattern_len)) < 1)
                error("UNIFORM: Length not parsed", 1);

            // Read the stride
            char *stride = strtok(NULL, ":");
            ssize_t strideval = 0;
            if (!stride)
                error("UNIFORM: Stride not found", 1);
            if (sscanf(stride, "%zd", &strideval) < 1)
                error("UNIFORM: Stride not parsed", 1);

            char *delta = strtok(NULL, ":");
            if (delta)
            {
                if (!strcmp(delta, "NR"))
                {
                    rc->delta = strideval*rc->pattern_len;
                    rc->deltas[0] = rc->delta;
                    rc->deltas_len = 1;
                }
                else
                {
                    if (sscanf(delta, "%zd", &(rc->delta)) < 1)
                        error("UNIFORM: delta not parsed", 1);
                    rc->deltas[0] = rc->delta;
                    rc->deltas_len = 1;
                }
            }

            for (int i = 0; i < rc->pattern_len; i++)
                rc->pattern[i] = i*strideval;
        }

        //LAPLACIAN:DIM:ORDER:N
        else if (!strcmp(optarg, "LAPLACIAN"))
        {
            int dim_val, order_val, problem_size_val;

            rc->type = LAPLACIAN;

            // Read the dimension
            char *dim = strtok(arg,":");
            if (!dim)
                error("LAPLACIAN: Dimension not found", 1);
            if (sscanf(dim, "%d", &dim_val) < 1)
                error("LAPLACIAN: Dimension not parsed", 1);

            // Read the order
            char *order = strtok(NULL, ":");
            if (!order)
                error("LAPLACIAN: Order not found", 1);
            if (sscanf(order, "%d", &order_val) < 1)
                error("LAPLACIAN: Order not parsed", 1);

            // Read the problem size
            char *problem_size = strtok(NULL, ":");
            if (!problem_size)
                error("LAPLACIAN: Problem size not found", 1);
            if (sscanf(problem_size, "%d", &problem_size_val) < 1)
                error("LAPLACIAN: Problem size not parsed", 1);

            rc->delta = 1;
            rc->deltas[0] = rc->delta;
            rc->deltas_len = 1;

            laplacian(dim_val, order_val, problem_size_val, rc);
        }

        // Mostly Stride 1 Mode
        // Arguments: index_length:list_of_breaks:list_of_deltas
        // list_of_deltas should be length 1 or the same length as
        // list_of_breaks.
        // The elements of both lists should be nonnegative and
        // the the elements of list_of_breaks should be strictly less
        // than index_length
        else if (!strcmp(optarg, "MS1"))
        {
            rc->type = MS1;

            char *len = strtok(arg,":");
            char *breaks = strtok(NULL,":");
            char *gaps = strtok(NULL,":");

            size_t *ms1_breaks = sp_malloc(sizeof(size_t), MAX_PATTERN_LEN, ALIGN_CACHE);
            size_t *ms1_deltas = sp_malloc(sizeof(size_t), MAX_PATTERN_LEN, ALIGN_CACHE);
            size_t ms1_breaks_len = 0;
            size_t ms1_deltas_len = 0;

            // Parse index length
            sscanf(len, "%zu", &(rc->pattern_len));

            // Parse breaks
            char *ptr = strtok(breaks, ",");
            size_t read = 0;
            if (!ptr)
                error("MS1: Breaks missing", 1);
            if (sscanf(ptr, "%zu", &(ms1_breaks[read++])) < 1)
                error("MS1: Failed to parse first break", 1);

            while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN)
            {
                if (sscanf(ptr, "%zu", &(ms1_breaks[read++])) < 1)
                    error("MS1: Failed to parse breaks", 1);
            }

            ms1_breaks_len = read;

            if(!gaps)
            {
                printf("1\n");
                error("error", ERROR);
            }

            ptr = strtok(gaps, ",");
            read = 0;
            if (ptr)
            {
                if (sscanf(ptr, "%zu", &(ms1_deltas[read++])) < 1)
                    error("Failed to parse first delta", 1);

                while ((ptr = strtok(NULL, ",")) && read < MAX_PATTERN_LEN)
                {
                    if (sscanf(ptr, "%zu", &(ms1_deltas[read++])) < 1)
                        error("Failed to parse deltas", 1);
                }
            }
            else
                error("MS1: deltas missing",1);

            ms1_deltas_len = read;

            rc->pattern[0] = -1;
            size_t last = -1;
            ssize_t j;
            for (int i = 0; i < rc->pattern_len; i++)
            {
                if ((j=setincludes(i, ms1_breaks, ms1_breaks_len))!=-1)
                   rc->pattern[i] = last+ms1_deltas[ms1_deltas_len>1?j:0];
                else
                    rc->pattern[i] = last + 1;
                last = rc->pattern[i];
            }

            free(ms1_breaks);
            free(ms1_deltas);
        }
        else
            error("Unrecognized mode in -p argument", 1);
    }

    // CUSTOM mode means that the user supplied a single index buffer on the command line
    else
    {
	if (quiet_flag < 3)
        	printf("Parse P Custom Pattern: %s\n", optarg);
        rc->type = CUSTOM;
        char *delim = ",";
        char *ptr = strtok(optarg, delim);
        size_t read = 0;
        if (!ptr)
            error("Pattern not found", 1);

        spIdx_t *mypat;

        mypat = sp_malloc(sizeof(spIdx_t), MAX_PATTERN_LEN, ALIGN_CACHE);

        if (sscanf(ptr, "%zu", &(mypat[read++])) < 1)
            error("Failed to parse first pattern element in custom mode", 1);

        while ((ptr = strtok(NULL, delim)) && read < MAX_PATTERN_LEN)
        {
            if (sscanf(ptr, "%zu", &(mypat[read++])) < 1)
                error("Failed to parse pattern", 1);
        }
        rc->pattern = mypat;
        rc->pattern_len = read;
    }

    if (rc->pattern_len == 0)
        error("Pattern length of 0", ERROR);

    if (rc->type == INVALID_IDX)
        error("No pattern type set", ERROR);
}

ssize_t setincludes(size_t key, size_t* set, size_t set_len)
{
    for (size_t i = 0; i < set_len; i++)
    {
        if (set[i] == key)
            return i;
    }
    return -1;
}

void print_run_config(struct run_config rc)
{
    printf("Index: %zu ", rc.pattern_len);
    printf("[");
    for (size_t i = 0; i < rc.pattern_len; i++)
    {
        printf("%zu", rc.pattern[i]);
        if (i != rc.pattern_len-1)
            printf(" ");
    }
    printf("]\n");
    if (rc.deltas_len > 0)
    {
        printf("Deltas: %zu ", rc.deltas_len);
        printf("[");
        for (size_t i = 0; i < rc.deltas_len; i++)
        {
            printf("%zu", rc.deltas[i]);
            if (i != rc.deltas_len-1)
                printf(" ");
        }
        printf("]\n");
        printf("Deltas_ps: %zu ", rc.deltas_len);
        printf("[");
        for (size_t i = 0; i < rc.deltas_len; i++)
        {
            printf("%zu", rc.deltas_ps[i]);
            if (i != rc.deltas_len-1)
                printf(" ");
        }
        printf("] (%zu)\n", rc.delta);
    }
    else
        printf("Delta: %zu\n", rc.delta);

    printf("kern: %s\n", kernel_name);
    printf("genlen: %zu\n", rc.generic_len);
}

void error(char *what, int code)
{
    if (code == ERROR)
        fprintf(err_file, "Error: ");
    else if (code == WARN)
    {
        if (verbose)
            fprintf(err_file, "Warning: ");
    }

    if (verbose || code)
    {
        fprintf(err_file, "%s", what);
        fprintf(err_file, "\n");
    }

    if(code)
        exit(code);
}

void safestrcopy(char *dest, const char *src)
{
    dest[0] = '\0';
    strncat(dest, src, STRING_SIZE-1);
}

int compare_ssizet(const void *a, const void *b)
{
    if (*(ssize_t*)a > *(ssize_t*)b) return 1;
    else if (*(ssize_t*)a < *(ssize_t*)b) return -1;
    else return 0;
}

void copy4(ssize_t *dest, ssize_t *a, int *off)
{
    for (int i = 0; i < 4; i++) {
        dest[i + *off] = a[i];
    }
    *off += 4;
}

void add4(ssize_t *dest, ssize_t *a, ssize_t *b, int *off)
{
    for (int i = 0; i < 4; i++) {
        dest[i + *off] = a[i] + b[i];
    }
    *off += 4;
}

void xkp_pattern(size_t *pat_, size_t dim)
{
    ssize_t pat[73];
    for (int i = 0; i < 73; i++) {
        pat[i] = i;
    }

    ssize_t Xp[4];
    ssize_t Xn[4];
    ssize_t Yp[4];
    ssize_t Yn[4];
    ssize_t Zp[4];
    ssize_t Zn[4];

    Xp[0] =  1; Xp[1] =  2; Xp[2] =  3; Xp[3] =  4;
    Xn[0] = -1; Xn[1] = -2; Xn[2] = -3; Xn[3] = -4;
    Yp[0] = dim;  Yp[1] =  2*dim; Yp[2] =  3*dim; Yp[3] =  4*dim;
    Yn[0] = -dim; Yn[1] = -2*dim; Yn[2] = -3*dim; Yn[3] = -4*dim;
    Zp[0] = dim*dim;  Zp[1] =  2*dim*dim; Zp[2] =  3*dim*dim; Zp[3] =  4*dim*dim;
    Zn[0] = -dim*dim; Zn[1] = -2*dim*dim; Zn[2] = -3*dim*dim; Zn[3] = -4*dim*dim;

    int idx = 0;
    pat[idx++] = 0;
    copy4(pat, Xp, &idx);
    copy4(pat, Xn, &idx);
    copy4(pat, Yp, &idx);
    copy4(pat, Yn, &idx);
    copy4(pat, Zp, &idx);
    copy4(pat, Zn, &idx);

    add4(pat, Xp, Yp, &idx);
    add4(pat, Xp, Zp, &idx);
    add4(pat, Xp, Yn, &idx);
    add4(pat, Xp, Zn, &idx);

    add4(pat, Xn, Yp, &idx);
    add4(pat, Xn, Zp, &idx);
    add4(pat, Xn, Yn, &idx);
    add4(pat, Xn, Zn, &idx);

    add4(pat, Yp, Zp, &idx);
    add4(pat, Yp, Zn, &idx);
    add4(pat, Yn, Zp, &idx);
    add4(pat, Yn, Zn, &idx);

    qsort(pat, 73, sizeof(ssize_t), compare_ssizet);

    ssize_t min = pat[0];
    for (int i = 1; i < 73; i++) {
        if (pat[i] < min) {
            min = pat[i];
        }
    }

    for (int i = 0; i < 73; i++) {
        pat[i] -= min;
    }

    for (int i = 0; i < 73; i++) {
        pat_[i] = pat[i];
    }

}
