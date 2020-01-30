#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
//#include "ocl-kernel-gen.h"
#include "parse-args.h"
#include "sgtype.h"
#include "sgbuf.h"
#include "sgtime.h"
#include "trace-util.h"
#include "sp_alloc.h"
#include "morton.h"

#if defined( USE_OPENCL )
	#include "../opencl/ocl-backend.h"
#endif
#if defined( USE_OPENMP )
	#include <omp.h>
	#include "openmp/omp-backend.h"
	#include "openmp/openmp_kernels.h"
#endif
#if defined ( USE_CUDA )
    #include <cuda.h>
    #include "cuda/cuda-backend.h"
#endif
#if defined( USE_SERIAL )
	#include "serial/serial-kernels.h"
#endif

#if defined( USE_PAPI )
    #include <papi.h>
    #include "papi_helper.h"
#endif

#define ALIGNMENT (4096)

#define xstr(s) str(s)
#define str(s) #s

const char* SPATTER_VERSION="0.4";

//SGBench specific enums
enum sg_backend backend = INVALID_BACKEND;

//Strings defining program behavior
char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];

int cuda_dev = -1;
int validate_flag = 0, quiet_flag = 0;
int aggregate_flag = 1;
int compress_flag = 0;
int papi_nevents = 0;
int stride_kernel = -1;
#ifdef USE_PAPI
char papi_event_names[PAPI_MAX_COUNTERS][STRING_SIZE];
int papi_event_codes[PAPI_MAX_COUNTERS];
long long papi_event_values[PAPI_MAX_COUNTERS];
extern const char* const papi_ctr_str[];
#endif

void print_papi_names() {
#ifdef USE_PAPI
    printf("\nPAPI Counters: %d\n", papi_nevents);
    if (papi_nevents > 0) {
        printf("{ ");
        for (int i = 0; i < papi_nevents; i++) {
            printf("\"%s\":\"%s\"", papi_ctr_str[i], papi_event_names[i]);
            if (i != papi_nevents-1) {
                printf(",\n  ");
            }
        }
        printf(" }\n");
    }
#endif
}
void print_system_info(){

    printf("\nRunning Spatter version %s\n",SPATTER_VERSION);
    printf("Compiler: %s ver. %s\n", xstr(SPAT_C_NAME), xstr(SPAT_C_VER));
    printf("Compiler Location: %s\n", xstr(SPAT_C));
    //printf("Contributors: Patrick Lavin, Jeff Young, Aaron Vose\n");
    printf("Backend: ");

    if(backend == OPENMP) printf("OPENMP\n");
    if(backend == OPENCL) printf("OPENCL\n");
    if(backend == CUDA) printf("CUDA\n");


    printf("Aggregate Results? %s\n", aggregate_flag ? "YES" : "NO");
#ifdef USE_CUDA
    if (backend == CUDA) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cuda_dev);
        printf("Device: %s\n", prop.name);
    }
#endif
    print_papi_names();

    printf("\n");
}

void print_header(){
    //printf("kernel op time source_size target_size idx_len bytes_moved actual_bandwidth omp_threads vector_len block_dim shmem\n");
    printf("%-7s %-12s %-12s", "config", "time(s)","bw(MB/s)");

#ifdef USE_PAPI
    for (int i = 0; i < papi_nevents; i++) {
        printf(" %-12s", papi_ctr_str[i]);
    }
#endif
    printf("\n");

}

int compare (const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}

/** Time reported in seconds, sizes reported in bytes, bandwidth reported in mib/s"
 */
double report_time(int ii, double time,  struct run_config rc, int idx){
    size_t bytes_moved = 0;
    double actual_bandwidth = 0;

    bytes_moved = sizeof(sgData_t) * rc.pattern_len * rc.generic_len;
    actual_bandwidth = bytes_moved / time / 1000. / 1000.;
    printf("%-7d %-12.4g %-12.6g", ii, time, actual_bandwidth);
#ifdef USE_PAPI
    for (int i = 0; i < papi_nevents; i++) {
        printf(" %-12lld", rc.papi_ctr[idx][i]);
    }
#endif
    printf("\n");
    return actual_bandwidth;
}


void report_time2(struct run_config* rc, int nrc) {
    double *bw = (double*)malloc(sizeof(double)*nrc);
    assert(bw);
    for (int k = 0; k < nrc; k++) {
        if (aggregate_flag) {
            double min_time_ms = rc[k].time_ms[0];
            int min_idx = 0;
            for (int i = 1; i < rc[k].nruns; i++) {
                if (rc[k].time_ms[i] < min_time_ms) {
                    min_time_ms = rc[k].time_ms[i];
                    min_idx = i;
                }
            }
            bw[k] = report_time(k, min_time_ms/1000., rc[k], min_idx);
        }
        else {
            for (int i = 0; i < rc[k].nruns; i++) {
                report_time(k, rc[k].time_ms[i]/1000., rc[k], i);
            }
        }
    }
    if (aggregate_flag) {
        double min = bw[0];
        double max = bw[0];
        double hmean = 0;
        double first, med, third;

        qsort(bw, nrc, sizeof(double), compare);

        for (int i = 0; i < nrc; i++) {
            if (bw[i] < min) {
                min = bw[i];
            }
            if (bw[i] > max) {
                max = bw[i];
            }
        }

        first = bw[nrc/4];
        med = bw[nrc/2];
        third = bw[3*nrc/4];

        // Harmonic mean
        for (int i = 0; i < nrc; i++) {
            hmean += 1./bw[i];
        }
        hmean = 1./hmean * nrc;

        // Harmonic Standard Error
        // Reference: The Standard Errors of the Geometric and
        // Harmonic Means and Their Application to Index Numbers
        // Author: Nilan Norris
        // URL: https://www.jstor.org/stable/2235723
        double E1_x = 0;
        for (int i = 0; i < nrc; i++) {
            E1_x += 1./bw[i];
        }
        E1_x = E1_x / nrc;

        double theta_22 = pow(1./E1_x, 2);

        double sig_1x = 0;
        for (int i = 0; i < nrc; i++) {
            sig_1x += pow(1./bw[i] - E1_x,2);
        }
        sig_1x = sqrt(sig_1x / nrc);

        double hstderr = theta_22 * sig_1x / sqrt(nrc);

        printf("\n%-11s %-12s %-12s %-12s %-12s\n", "Min", "25%","Med","75%", "Max");
        printf("%-12.6g %-12.6g %-12.6g %-12.6g %-12.6g\n", min, first, med, third, max);
        printf("%-12s %-12s\n", "H.Mean", "H.StdErr");
        printf("%-12.6g %-12.6g\n", hmean, hstderr);
        /*
        printf("%.3lf\t%.3lf\n", hmean, stddev);
        */
    }
    free(bw);

}

void print_data(double *buf, size_t len){
    for (size_t i = 0; i < len; i++){
        printf("%.0lf ", buf[i]);
    }
    printf("\n");
}
void print_sizet(size_t *buf, size_t len){
    for (size_t i = 0; i < len; i++){
        printf("%zu ", buf[i]);
    }
    printf("\n");
}

void emit_configs(struct run_config *rc, int nconfigs);
uint64_t isqrt(uint64_t x);
uint64_t icbrt(uint64_t x);


int main(int argc, char **argv)
{

    // =======================================
    // Declare Variables
    // =======================================

    // source and target are used for the gather and scatter operations.
    // data is gathered from source and placed into target
    sgDataBuf  source;
    sgDataBuf  target;

    // OpenCL Specific
    #ifdef USE_OPENCL
    size_t global_work_size = 1;
    char   *kernel_string;
    cl_uint work_dim = 1;
    #endif

    // =======================================
    // Parse Command Line Arguments
    // =======================================

    struct run_config *rc;
    int nrc = 0;
    parse_args(argc, argv, &nrc, &rc);

    if (nrc <= 0) {
        error("No run configurations parsed", ERROR);
    }

    // If indices span many pages, compress them so that there are no
    // pages in the address space which are never accessed
    // Pages are assumed to be 4KiB
    if (compress_flag) {
        for (int i = 0; i < nrc; i++) {
            compress_indices(rc[i].pattern, rc[i].pattern_len);
        }
    }

    struct run_config *rc2 = rc;

    // Allocate space for timing and papi counter information

    for (int i = 0; i < nrc; i++) {
        rc2[i].time_ms = (double*)malloc(sizeof(double) * rc2[i].nruns);
#ifdef USE_PAPI
        rc2[i].papi_ctr = (long long **)malloc(sizeof(long long *) * rc2[i].nruns);
        for (int j = 0; j < rc2[i].nruns; j++){
            rc2[i].papi_ctr[j] = (long long*)malloc(sizeof(long long) * papi_nevents);
        }
#endif
    }
    // =======================================
    // Initialize PAPI Library
    // =======================================

#ifdef USE_PAPI
    // Powering up a space shuttle probably has fewer checks than initlizing papi
    int err = PAPI_library_init(PAPI_VER_CURRENT);
    if (err !=PAPI_VER_CURRENT && err > 0) {
        error ("PAPI library version mismatch", ERROR);
    }
    if (err < 0) papi_err(err, __LINE__, __FILE__);
    err = PAPI_is_initialized();
    if (err != PAPI_LOW_LEVEL_INITED) {
        error ("PAPI was not initialized", ERROR);
    }

    // OK, now that papi is finally inizlized, we need to make our EventSet
    // First, convert names to codes
    for (int i = 0; i < papi_nevents; i++) {
        papi_err(PAPI_event_name_to_code(papi_event_names[i],&papi_event_codes[i]), __LINE__, __FILE__);
    }

    int EventSet = PAPI_NULL;
    papi_err(PAPI_create_eventset(&EventSet), __LINE__, __FILE__);
    for (int i = 0; i < papi_nevents; i++) {
        papi_err(PAPI_add_event(EventSet, papi_event_codes[i]), __LINE__, __FILE__);
    }

#endif

    // =======================================
    // Initialize OpenCL Backend
    // =======================================

    /* Create a context and corresponding queue */
    #ifdef USE_OPENCL
    if (backend == OPENCL) {
    	initialize_dev_ocl(platform_string, device_string);
    }
    #endif

    // =======================================
    // Compute Buffer Sizes
    // =======================================

    if (rc2[0].kernel != GATHER && rc2[0].kernel != SCATTER) {
        printf("Error: Unsupported kernel\n");
        exit(1);
    }
    size_t max_source_size = 0;
    size_t max_target_size = 0;
    size_t max_pat_len = 0;
    size_t max_ptrs = 0;
    size_t max_morton = 0;
    for (int i = 0; i < nrc; i++) {

        size_t max_pattern_val = rc2[i].pattern[0];
        for (size_t j = 0; j < rc2[i].pattern_len; j++) {
            if (rc2[i].pattern[j] > max_pattern_val) {
                max_pattern_val = rc2[i].pattern[j];
            }
        }

        //printf("count: %zu, delta: %zu, %zu\n", rc2[i].generic_len, rc2[i].delta, rc2[i].generic_len*rc2[i].delta);

        size_t cur_source_size = ((max_pattern_val + 1) + (rc2[i].generic_len-1)*rc2[i].delta) * sizeof(sgData_t);
        //printf("max_pattern_val: %zu, source_size %zu\n", max_pattern_val, cur_source_size);
        if (cur_source_size > max_source_size) {
            max_source_size = cur_source_size;
        }

        size_t cur_target_size = rc2[i].pattern_len * sizeof(sgData_t) * rc2[i].wrap;
        if (cur_target_size > max_target_size) {
            max_target_size = cur_target_size;
        }

        if (rc2[i].omp_threads > max_ptrs) {
            max_ptrs = rc2[i].omp_threads;
        }

        if (rc2[i].pattern_len > max_pat_len) {
            max_pat_len = rc2[i].pattern_len;
        }

        if (rc2[i].morton == 1) {
            rc2[i].morton_order = z_order_1d(rc2[i].generic_len, rc2[i].morton_block);
        } else if (rc2[i].morton == 2) {
            rc2[i].morton_order = z_order_2d(isqrt(rc2[i].generic_len), rc2[i].morton_block);
        } else if (rc2[i].morton == 3) {
            rc2[i].morton_order = z_order_3d(icbrt(rc2[i].generic_len), rc2[i].morton_block);
        }

        if (rc2[i].morton) {
            if (rc2[i].generic_len > max_morton) {
                max_morton = rc2[i].generic_len;
            }
        }
    }

    source.size = max_source_size;
    source.len = source.size / sizeof(sgData_t);

    target.size = max_target_size;
    target.len = target.size / sizeof(sgData_t);

    target.nptrs = max_ptrs;

    // =======================================
    // Create OpenCL Kernel
    // =======================================
    #ifdef USE_OPENCL
    if (backend == OPENCL) {
        //kernel_string = ocl_kernel_gen(index_len, vector_len, kernel);
        kernel_string = read_file(kernel_file);
        sgp = kernel_from_string(context, kernel_string, kernel_name, NULL);
        if (kernel_string) {
            free(kernel_string);
        }
    }
    #endif

    // =======================================
    // Create Host Buffers, Fill With Data
    // =======================================
    source.host_ptr = (sgData_t*) sp_malloc(source.size, 1, ALIGN_CACHE);

    // replicate the target space for every thread
    target.host_ptrs = (sgData_t**) sp_malloc(sizeof(sgData_t*), target.nptrs, ALIGN_CACHE);
    for (size_t i = 0; i < target.nptrs; i++) {
        target.host_ptrs[i] = (sgData_t*) sp_malloc(target.size, 1, ALIGN_PAGE);
    }
    //    printf("-- here -- \n");

    // Populate buffers on host
    #pragma omp parallel for
    for (int i = 0; i < source.len; i++) {
        source.host_ptr[i] = i % (source.len / 64);
    }
    random_data(source.host_ptr, source.len);

    // =======================================
    // Create Device Buffers, Transfer Data
    // =======================================
    #ifdef USE_OPENCL
    if (backend == OPENCL) {
        //TODO: Rewrite to not take index buffers
        //create_dev_buffers_ocl(&source, &target, &si, &ti);
    }
    #endif

    #ifdef USE_CUDA
    sgIdx_t *pat_dev;
    uint32_t *order_dev;
    if (backend == CUDA) {
        //TODO: Rewrite to not take index buffers
        create_dev_buffers_cuda(&source);
        cudaMalloc((void**)&pat_dev, sizeof(sgIdx_t) * max_pat_len);
        cudaMalloc((void**)&order_dev, sizeof(uint32_t) * max_morton);
        cudaMemcpy(source.dev_ptr_cuda, source.host_ptr, source.size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    #endif


    // =======================================
    // Execute Benchmark
    // =======================================

    // Print some header info
    /*
    if (print_header_flag)
    {
        print_system_info();
        emit_configs(rc2, nrc);
        print_header();
    }
    */
    if (quiet_flag < 1) {
        print_system_info();
    }
    if (quiet_flag < 2) {
        emit_configs(rc2, nrc);
    }
    if (quiet_flag < 3) {
        print_header();
    }



    // Print config info

    for (int k = 0; k < nrc; k++) {
        // Time OpenCL Kernel
        #ifdef USE_OPENCL
        if (backend == OPENCL) {

        }
        #endif // USE_OPENCL

        // Time CUDA Kernel
        #ifdef USE_CUDA
        int wpt = 1;
        if (backend == CUDA) {
            float time_ms = 2;
            for (int i = -10; i < (int)rc2[k].nruns; i++) {
#define arr_len (1)
                unsigned long global_work_size = rc2[k].generic_len / wpt * rc2[k].pattern_len;
                unsigned long local_work_size = rc2[k].local_work_size;
                unsigned long grid[arr_len]  = {global_work_size/local_work_size};
                unsigned long block[arr_len] = {local_work_size};
                if (rc2[k].random_seed == 0) {
                    time_ms = cuda_block_wrapper(arr_len, grid, block, rc2[k].kernel, source.dev_ptr_cuda, pat_dev, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap, wpt, rc2[k].morton, rc2[k].morton_order, order_dev, rc[k].stride_kernel);
                } else {
                    time_ms = cuda_block_random_wrapper(arr_len, grid, block, rc2[k].kernel, source.dev_ptr_cuda, pat_dev, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap, wpt, rc2[k].random_seed);
                }

                if (i>=0) rc2[k].time_ms[i] = time_ms;
            }


        }

        #endif // USE_CUDA



        // Time OpenMP Kernel
        #ifdef USE_OPENMP
        if (backend == OPENMP) {
            omp_set_num_threads(rc2[k].omp_threads);

            // Start at -1 to do a cache warm
            for (int i = -1; i < (int)rc2[k].nruns; i++) {

                if (i!=-1) sg_zero_time();
#ifdef USE_PAPI
                if (i!=-1) profile_start(EventSet, __LINE__, __FILE__);
#endif

                switch (rc2[k].kernel) {
                    case SG:
                        if (rc2[k].op == OP_COPY) {
                            //sg_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr,index_len);
                        } else {
                            //sg_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case SCATTER:
                        if (rc2[k].random_seed >= 1) {
                            scatter_smallbuf_random(source.host_ptr, target.host_ptrs, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap, rc2[k].random_seed);
                        }
                        else if (rc2[k].op == OP_COPY) {
                            scatter_smallbuf(source.host_ptr, target.host_ptrs, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap);
                            // scatter_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        } else {
                            // scatter_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case GATHER:
                        if (rc2[k].random_seed >= 1) {
                            gather_smallbuf_random(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap, rc2[k].random_seed);
                        }
                        else if (rc2[k].deltas_len <= 1) {
                            if (rc2[k].morton) {
                                gather_smallbuf_morton(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap, rc2[k].morton_order);
                            } else {
                                gather_smallbuf(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap);
                            }
                        } else {
                            gather_smallbuf_multidelta(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].deltas_ps, rc2[k].generic_len, rc2[k].wrap, rc2[k].deltas_len);
                        }
                        break;
                    default:
                        printf("Error: Unable to determine kernel\n");
                        break;
                }

#ifdef USE_PAPI
                if (i!= -1) profile_stop(EventSet, rc2[k].papi_ctr[i], __LINE__, __FILE__);
#endif
                if (i!= -1) rc2[k].time_ms[i] = sg_get_time_ms();

            }

            //report_time2(rc2, nrc);
        }
        #endif // USE_OPENMP

        // Time Serial Kernel
        #ifdef USE_SERIAL
        if (backend == SERIAL) {

            for (int i = 0; i <= rc2[k].nruns; i++) {

                if (i!=-1) sg_zero_time();
#ifdef USE_PAPI
                if (i!=-1) profile_start(EventSet, __LINE__, __FILE__);
#endif

                //TODO: Rewrite serial kernel
                switch (rc2[k].kernel) {
                    case SCATTER:
                        scatter_smallbuf_serial(source.host_ptr, target.host_ptrs, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap);
                        break;
                    case GATHER:
                        gather_smallbuf_serial(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap);
                        break;
                    default:
                        printf("Error: Unable to determine kernel\n");
                        break;
                }

                //double time_ms = sg_get_time_ms();
                //if (i!=0) report_time(k, time_ms/1000., rc2[k], i);
#ifdef USE_PAPI
                if (i!= -1) profile_stop(EventSet, rc2[k].papi_ctr[i], __LINE__, __FILE__);
#endif
                if (i!= -1) rc2[k].time_ms[i] = sg_get_time_ms();
            }
        }
        #endif // USE_SERIAL
    }

    report_time2(rc2, nrc);

#ifdef USE_CUDA
    cudaMemcpy(source.host_ptr, source.dev_ptr_cuda, source.size, cudaMemcpyDeviceToHost);
#endif
    int good = 0;
    int bad  = 0;
    for (int i = 0; i < source.len; i++) {
        if (source.host_ptr[i] == 1337.) {
            good++;
        }else {
            bad++;
        }
    }
    //printf("\ngood: %d, bad: %d\n", good, bad);


    // =======================================
    // Validation
    // =======================================
    if(validate_flag) {

        //TODO: Rewrite validataion
/*
#ifdef USE_OPENCL
        if (backend == OPENCL) {
            clEnqueueReadBuffer(queue, target.dev_ptr_opencl, 1, 0, target.size,
                target.host_ptr, 0, NULL, &e);
            clWaitForEvents(1, &e);
        }
#endif

#ifdef USE_CUDA
        if (backend == CUDA) {
            cudaError_t cerr;
            cerr = cudaMemcpy(target.host_ptr, target.dev_ptr_cuda, target.size, cudaMemcpyDeviceToHost);
            if(cerr != cudaSuccess){
                printf("transfer failed\n");
            }
            cudaDeviceSynchronize();
        }
#endif

        sgData_t *target_backup_host = (sgData_t*) sg_safe_cpu_alloc(target.size);
        memcpy(target_backup_host, target.host_ptr, target.size);

    // =======================================
	VALIDATION
       =======================================
    //
    if(validate_flag) {
#ifdef USE_OPENCL
        if (backend == OPENCL) {
            clEnqueueReadBuffer(queue, target.dev_ptr_opencl, 1, 0, target.size,
                target.host_ptr, 0, NULL, &e);
            clWaitForEvents(1, &e);
        }
#endif

#ifdef USE_CUDA
        if (backend == CUDA) {
            cudaError_t cerr;
            cerr = cudaMemcpy(target.host_ptr, target.dev_ptr_cuda, target.size, cudaMemcpyDeviceToHost);
            if(cerr != cudaSuccess){
                printf("transfer failed\n");
            }
            cudaDeviceSynchronize();
        }
#endif

        sgData_t *target_backup_host = (sgData_t*) sg_safe_cpu_alloc(target.size);
        memcpy(target_backup_host, target.host_ptr, target.size);

    	// TODO: Issue - 13: Replace the hard-coded execution of each function with calls to the serial backend
        switch (kernel) {
            case SG:
                for (size_t i = 0; i < index_len; i++){
                    target.host_ptr[ti.host_ptr[i]] = source.host_ptr[si.host_ptr[i]];
                }
                break;
            case SCATTER:
                for (size_t i = 0; i < index_len; i++){
                    target.host_ptr[ti.host_ptr[i]] = source.host_ptr[i];
                }
                break;
            case GATHER:
                for (size_t i = 0; i < index_len; i++){
                    target.host_ptr[i] = source.host_ptr[si.host_ptr[i]];
                }
                break;
        }


        int num_err = 0;
        for (size_t i = 0; i < target.len; i++) {
            if (target.host_ptr[i] != target_backup_host[i]) {
                printf("%zu: host %lf, device %lf\n", i, target.host_ptr[i], target_backup_host[i]);
                num_err++;
            }
            if (num_err > 99) {
                printf("Too many errors. Exiting.\n");
                exit(1);
            }
        }
        */
    //}
    }

    // Free Memory
    free(source.host_ptr);
    for (size_t i = 0; i < target.nptrs; i++) {
      free(target.host_ptrs[i]);
    }
    if (target.nptrs != 0) {
      free(target.host_ptrs);
    }

    for (int i = 0; i < nrc; i++) {
        if (rc2[i].morton_order) {
            free(rc2[i].morton_order);
        }
        free(rc2[i].time_ms);
#ifdef USE_PAPI
        for (int j = 0; j < rc2[i].nruns; j++){
            free(rc2[i].papi_ctr[j]);
        }
        free(rc2[i].papi_ctr);
#endif
    }

  free(rc);
  //printf("Mem used: %lld MiB\n", get_mem_used()/1024/1024);
}

void emit_configs(struct run_config *rc, int nconfigs)
{

    printf("Run Configurations\n");

    printf("[ ");
    for (int i = 0; i < nconfigs; i++) {
        if (i != 0) {
            printf("  ");
        }

        printf("{");

        // Pattern Type
        printf("\'name\':\'%s\', ", rc[i].name);

        // Kernel
        switch (rc[i].kernel) {
        case GATHER:
            printf("\'kernel\':\'Gather\', ");
            break;
        case SCATTER:
            printf("\'kernel\':\'Scatter\', ");
            break;
        case SG:
            printf("\'kernel\':\'GS\', ");
            break;
        case INVALID_KERNEL:
            error ("Invalid kernel sent to emit_configs", ERROR);
            break;
        }

        // Pattern
        printf("\'pattern\':[");
        for (int j = 0; j < rc[i].pattern_len; j++) {
            printf("%zu", rc[i].pattern[j]);
            if (j != rc[i].pattern_len-1) {
                printf(",");
            }
        }
        printf("], ");

        //Delta
        //TODO: multidelta
        if (rc[i].deltas_len == 1) {
            printf("\'delta\':%zd", rc[i].delta);
        } else {
            printf("\'deltas\':[");
            for (int j = 0; j < rc[i].deltas_len; j++) {
                printf("%zu", rc[i].deltas[j]);
                if (j != rc[i].deltas_len-1) {
                    printf(",");
                }
            }
            printf("]");

        }
        printf(", ");

        // Len
        printf("\'length\':%zu, ", rc[i].generic_len);

        if (rc[i].random_seed > 0) {
            printf("\'seed\':%zu, ", rc[i].random_seed);
        }

        // Aggregate
        if (aggregate_flag) {
            printf("\'agg\':%zu, ", rc[i].nruns);
        }

        // Wrap
        if (aggregate_flag) {
            printf("\'wrap\':%zu, ", rc[i].wrap);
        }

        // OpenMP Threads
        if (backend == OPENMP) {
            printf("\'threads\':%zu", rc[i].omp_threads);
        }

        // OpenMP Threads
        if (rc[i].stride_kernel!=-1) {
            printf("\'stride_kernel\':%d", rc[i].stride_kernel);
        }

        // Morton
        if (rc[i].morton) {
            printf(", \'morton\':%d", rc[i].morton);
        }

        if (rc[i].morton) {
            printf(", \'mblock\':%d", rc[i].morton_block);
        }

        printf("}");

        if (i != nconfigs-1) {
            printf(",\n");
        }

    }
    printf(" ]\n\n");
}

// From http://www.codecodex.com/wiki/Calculate_an_integer_square_root
uint64_t isqrt(uint64_t x)
{
    uint64_t op, res, one;

    op = x;
    res = 0;

    /* "one" starts at the highest power of four <= than the argument. */
    one = 1 << 30;  /* second-to-top bit set */
    while (one > op) one >>= 2;

    while (one != 0) {

        if (op >= res + one) {
            op -= res + one;
            res += one << 1;  // <-- faster than 2 * one
        }
        res >>= 1;
        one >>= 2;
    }
    return res;
}

// From https://gist.github.com/anonymous/729557
uint64_t icbrt(uint64_t x) {
    int s;
    uint64_t y;
    uint64_t b;
    y = 0;
    for (s = 63; s >= 0; s -= 3) {
        y += y;
        b = 3*y*((uint64_t) y + 1) + 1;
        if ((x >> s) >= b) {
                x -= b << s;
                y++;
        }
    }
    return y;
}
