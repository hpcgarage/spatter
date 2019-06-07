#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
//#include "ocl-kernel-gen.h"
#include "parse-args.h"
#include "sgtype.h"
#include "sgbuf.h"
#include "sgtime.h"
#include "trace-util.h"

#if defined( USE_OPENCL )
	#include "../opencl/ocl-backend.h"
#endif
#if defined( USE_OPENMP )
	#include <omp.h>
	#include "../openmp/omp-backend.h"
	#include "../openmp/openmp_kernels.h"
#endif
#if defined ( USE_CUDA )
    #include <cuda.h>
    #include "../cuda/cuda-backend.h"
#endif
#if defined( USE_SERIAL )
	#include "../serial/serial-kernels.h"
#endif

#define ALIGNMENT (64)


//SGBench specific enums
enum sg_backend backend = INVALID_BACKEND;
enum sg_kernel  kernel  = INVALID_KERNEL;
enum sg_op      op      = OP_COPY;

//Strings defining program behavior
char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];
char config_file[STRING_SIZE];

size_t source_len;
size_t target_len;
size_t index_len;
size_t generic_len = 0;
size_t seed;
size_t wrap = 1;
size_t R = 10;
size_t workers = 1;
size_t vector_len = 1;
size_t local_work_size = 1;
size_t ms1_gap = 0; 
size_t ms1_run = 0;
ssize_t us_stride = 0; 
ssize_t us_delta = 0;

unsigned int shmem = 0;
int random_flag = 0;
int ms1_flag = 0;
int config_flag = 0;
int custom_flag = 0;

int json_flag = 0, validate_flag = 0, print_header_flag = 1;

// NOIDX MODE OPTIONS
int noidx_flag            = 0;
int noidx_explicit_mode   = 0;
int noidx_predef_us_mode  = 0;
int noidx_predef_ms1_mode = 0;
int noidx_file_mode       = 0;
int noidx_onesided        = 0;

size_t noidx_pattern[MAX_PATTERN_LEN] = {0};
size_t noidx_pattern_len  = 0;
char  noidx_pattern_file[STRING_SIZE] = {0};

ssize_t noidx_delta       = -1;
ssize_t noidx_us_stride   = -1;
size_t noidx_ms1_deltas[MAX_PATTERN_LEN] =  {0};
size_t noidx_ms1_breaks[MAX_PATTERN_LEN] =  {0};
size_t noidx_ms1_deltas_len = 0;
size_t noidx_ms1_breaks_len = 0;
ssize_t noidx_ms1_delta   = -1;

size_t noidx_window;
size_t noidx_size;

int verbose = 0;

void print_system_info(){

    printf("Running Spatter version 0.0\n");
    printf("Backend: ");

    if(backend == OPENMP) printf("OPENMP ");
    if(backend == OPENCL) printf("OPENCL ");
    if(backend == CUDA) printf("CUDA ");

    printf("\n");

    if(noidx_flag) {
        printf("Index pattern: ");
    } 

    if (noidx_predef_ms1_mode) printf("MS1: [");
    if (noidx_predef_us_mode) printf("UNIFORM: [");
    if (noidx_explicit_mode) printf("CUSTOM: [");
        for (int i = 0; i < noidx_pattern_len; i++) {
            printf("%zu", noidx_pattern[i]);
            if (i != noidx_pattern_len-1) printf(" ");
        }
    printf("]\n\n");


}

void print_header(){
    if (noidx_flag) 
        printf("kernel op time source_size target_size idx_len bytes_moved actual_bandwidth omp_threads vector_len block_dim shmem\n");
    else
        printf("kernel op time source_size target_size idx_size bytes_moved usable_bandwidth actual_bandwidth nthreads vector_len block_dim shmem\n");
}

void *sg_safe_cpu_alloc (size_t size) {
    void *ptr = aligned_alloc (ALIGNMENT, size);
    if (!ptr) {
        printf("Falied to allocate memory on cpu\n");
        exit(1);
    }
    return ptr;
}

/** Time reported in seconds, sizes reported in bytes, bandwidth reported in mib/s"
 */
void report_time(double time, size_t source_size, size_t target_size, size_t index_size,  size_t vector_len){

    if(kernel == SCATTER) printf("SCATTER ");
    if(kernel == GATHER) printf("GATHER ");
    if(kernel == SG) printf("SG ");

    if(op == OP_COPY) printf("COPY ");
    if(op == OP_ACCUM) printf("ACCUM ");

    printf("%lf %zu %zu ", time, source_size, target_size);

    if (noidx_flag)
        printf("%zu ", noidx_pattern_len);
    else 
        printf("%zu ", index_size);

    size_t bytes_moved = 0;
    double usable_bandwidth = 0;
    double actual_bandwidth = 0;
    if (noidx_flag){
        bytes_moved = 2 * sizeof(sgData_t) * noidx_pattern_len * generic_len;
        usable_bandwidth = 0;
        actual_bandwidth = bytes_moved / time / 1000. / 1000.;
    } else {
        bytes_moved = 2 * index_len * sizeof(sgData_t);
        usable_bandwidth = bytes_moved / time / 1000. / 1000.;
        actual_bandwidth = (bytes_moved + index_size) / time / 1000. / 1000.;
    }
    if (noidx_flag) 
        printf("%zu %lf ", bytes_moved, actual_bandwidth);
    else
        printf("%zu %lf %lf ", bytes_moved, usable_bandwidth, actual_bandwidth);

    //How many threads were used - currently refers to CPU systems
    size_t worker_threads = workers;
    #ifdef USE_OPENMP
	worker_threads = omp_get_max_threads();
    #endif

    printf("%zu %zu %zu %u", worker_threads, vector_len, local_work_size, shmem);

    printf("\n");

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

int main(int argc, char **argv)
{

    sgDataBuf  source;
    sgDataBuf  target;
    sgIndexBuf si; //source_index
    sgIndexBuf ti; //target_index
    struct trace tr;

    size_t cpu_cache_size = 30720 * 1000; 
    size_t cpu_flush_size = cpu_cache_size * 8;
    #ifdef USE_OPENCL
	cl_ulong device_cache_size = 0;
    	cl_uint work_dim = 1;
    #endif
    size_t global_work_size = 1;

    size_t *pattern;
    
    char *kernel_string;

    /* Parse command line arguments */
    parse_args(argc, argv);

    if (config_flag) {
        read_trace(&tr, config_file);
        reweight_trace(tr);
    };

    /* =======================================
	Initalization
       =======================================
    */

    /* Create a context and corresponding queue */
    #ifdef USE_OPENCL
    if (backend == OPENCL) {
    	initialize_dev_ocl(platform_string, device_string);
    }
    #endif

    #ifdef USE_CUDA 
    if (backend == CUDA) {
        /*
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 1);

        printf("sm's: %d\n", prop.multiProcessorCount);
        */
    }
    #endif

    if (noidx_flag) {
        source.size = (noidx_window + (generic_len-1)*noidx_delta) * sizeof(double);
        target.size = generic_len * noidx_pattern_len * sizeof(double);
        source.len = source.size / sizeof(sgData_t);
        target.len = target.size / sizeof(sgData_t);
    } else {
        source.len = source_len;
        target.len = target_len;
        si.len     = index_len;
        ti.len     = index_len;

        /* These are the total size of the data allocated for each buffer */
        source.size = source.len * sizeof(sgData_t);
        target.size = target.len * sizeof(sgData_t);

        printf("source.len %zu, si.len %zu\n", source.len, si.len);
        si.stride = source.len / si.len;
        ti.stride = target.len / ti.len;

        if (kernel == GATHER || kernel == SG) {
            if(wrap > si.stride) {
                wrap = si.stride;
            }
        }
        if (kernel == SCATTER || kernel == SG) {
            if(wrap > ti.stride) { 
                wrap = ti.stride;
            }
        }

        if (wrap > 1){
            if (kernel == GATHER || kernel == SG) {
                source.size = source.size / wrap;
                source.len = source.len / wrap;
            }
            if (kernel == SCATTER || kernel == SG) {
                target.size = target.size / wrap;
                target.len = target.len / wrap;
            }
        }

       si.size     = si.len * sizeof(sgIdx_t);
       ti.size     = ti.len * sizeof(sgIdx_t);
    }

    /* Create the kernel */
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


    if (noidx_flag) {
        si.len = 0;
        ti.len = 0;
        si.size = 0;
        ti.size = 0;
    }

    if (!noidx_flag) {
        si.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(si.size); 
        ti.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(ti.size); 

        if (ms1_flag) {
            if (kernel == SCATTER) {
                linear_indices(si.host_ptr, si.len, 1, 1, 0);
                ms1_indices(ti.host_ptr, ti.len, 1, ms1_run, ms1_gap);
            }else if (kernel == GATHER) {
                ms1_indices(si.host_ptr, si.len, 1, ms1_run, ms1_gap);
                linear_indices(ti.host_ptr, ti.len, 1, 1, 0);
            } else {
                printf("MS1 pattern is only supported for scatter and gather\n");
                exit(1);
            }
        } else {
            if (wrap > 1) {
                wrap_indices(si.host_ptr, si.len, 1, si.stride, wrap);
                wrap_indices(ti.host_ptr, ti.len, 1, ti.stride, wrap);
            } else {
                linear_indices(si.host_ptr, si.len, 1, si.stride, random_flag);
                linear_indices(ti.host_ptr, ti.len, 1, ti.stride, random_flag);
            }
        }

        if (config_flag) {
            if (kernel == SCATTER) {
                size_t reqd_len = trace_indices(ti.host_ptr, ti.len, tr);
                target.len = reqd_len;
                target.size = target.len * sizeof(sgData_t);
            } else if (kernel == GATHER) {
                size_t reqd_len = trace_indices(si.host_ptr, si.len, tr);
                source.len = reqd_len;
                source.size = source.len * sizeof(sgData_t);
            } else {
                printf("Error: pattern files only support scatter and gather kernels\n");
            }
        }

    }

    /* Create buffers on host */
    source.host_ptr = (sgData_t*) sg_safe_cpu_alloc(source.size); 
    target.host_ptr = (sgData_t*) sg_safe_cpu_alloc(target.size); 
    /* Populate buffers on host */
    random_data(source.host_ptr, source.len);

    /*
    for(size_t kk = 0; kk < si.len; kk++){
        printf("%zu ", si.host_ptr[kk]);
    }
    printf("\n");
    */

    /* Create buffers on device and transfer data from host */
    #ifdef USE_OPENCL
    if (backend == OPENCL) {
        create_dev_buffers_ocl(&source, &target, &si, &ti);
    }
    #endif

    #ifdef USE_CUDA
    if (backend == CUDA) {
        create_dev_buffers_cuda(&source, &target, &si, &ti);
        cudaMemcpy(source.dev_ptr_cuda, source.host_ptr, source.size, cudaMemcpyHostToDevice);
        cudaMemcpy(si.dev_ptr_cuda, si.host_ptr, si.size, cudaMemcpyHostToDevice);
        cudaMemcpy(ti.dev_ptr_cuda, ti.host_ptr, ti.size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    #endif

    
    /* =======================================
	Benchmark Execution
       =======================================
    */

    /* Begin benchmark */
    if (print_header_flag) 
    {
        print_system_info();
        print_header();
    }
    

    /* Time OpenCL Kernel */
    #ifdef USE_OPENCL
    if (backend == OPENCL) {

        global_work_size = si.len / vector_len;
        assert(global_work_size > 0);
        cl_ulong start = 0, end = 0; 
        for (int i = 0; i <= R; i++) {
             
            start = 0; end = 0;

           cl_event e = 0; 

            SET_4_KERNEL_ARGS(sgp, target.dev_ptr_opencl, source.dev_ptr_opencl,
                    ti.dev_ptr_opencl, si.dev_ptr_opencl);

            CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, sgp, work_dim, NULL, 
                       &global_work_size, &local_work_size, 
                      0, NULL, &e)); 
            clWaitForEvents(1, &e);

            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));

            cl_ulong time_ns = end - start;
            double time_s = time_ns / 1000000000.;
            if (i!=0) report_time(time_s, source.size, target.size, si.size, vector_len);


        }

    }
    #endif // USE_OPENCL

    /* Time CUDA Kernel */
    #ifdef USE_CUDA
    if (backend == CUDA) {

        global_work_size = si.len / vector_len;
        long start = 0, end = 0; 
        for (int i = 0; i <= R; i++) {
             
            start = 0; end = 0;
#define arr_len (1) 
            unsigned int grid[arr_len]  = {global_work_size/local_work_size};
            unsigned int block[arr_len] = {local_work_size};
            
            float time_ms = cuda_sg_wrapper(kernel, vector_len, 
                    arr_len, grid, block, target.dev_ptr_cuda, source.dev_ptr_cuda, 
                   ti.dev_ptr_cuda, si.dev_ptr_cuda, shmem); 
            cudaDeviceSynchronize();

            double time_s = time_ms / 1000.;
            if (i!=0) report_time(time_s, source.size, target.size, si.size, vector_len);

        }

    }
    #endif // USE_CUDA



    /* Time OpenMP Kernel */
    #ifdef USE_OPENMP
    if (backend == OPENMP) {

        for (int i = 0; i <= R; i++) {

            sg_zero_time();

            switch (kernel) {
                case SG:
                    if (op == OP_COPY) 
                        sg_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    else 
                        sg_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                case SCATTER:
                    if (op == OP_COPY)
				scatter_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
	                        index_len);
                    else
                        scatter_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                case GATHER:
                    if (noidx_flag)
                        if (noidx_onesided) {
                            //printf(" -- onesided mode\n");
                            gather_stride_noidx_os(target.host_ptr, source.host_ptr, noidx_pattern, noidx_pattern_len, noidx_delta, generic_len, noidx_onesided);
                        } else {
                            //printf(" -- twosided mode\n");
                            gather_noidx(target.host_ptr, source.host_ptr, noidx_pattern, noidx_pattern_len, noidx_delta, generic_len);
                        }
                    else if (op == OP_COPY)
				gather_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
	                        index_len);
                    else
                        gather_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                default:
                    printf("Error: Unable to determine kernel\n");
                    break;
            }

            double time_ms = sg_get_time_ms();
            if (i!=0) report_time(time_ms/1000., source.size, target.size, si.size, vector_len);

        }
    }
    #endif // USE_OPENMP
    
    /* Time Serial Kernel */
    #ifdef USE_SERIAL
    if (backend == SERIAL) {

        for (int i = 0; i <= R; i++) {

            sg_zero_time();

            switch (kernel) {
                case SG:
                    if (op == OP_COPY) 
                        sg_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    else 
                        sg_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                case SCATTER:
                    if (op == OP_COPY)
				scatter_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
	                        index_len);
                    else
                        scatter_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                case GATHER:
                    if (op == OP_COPY)
				gather_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
	                        index_len);
                    else
                        gather_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len);
                    break;
                default:
                    printf("Error: Unable to determine kernel\n");
                    break;
            }

            double time_ms = sg_get_time_ms();
            if (i!=0) report_time(time_ms/1000., source.size, target.size, si.size, vector_len);
        }
    }
    #endif // USE_SERIAL
    

    /* =======================================
	VALIDATION
       =======================================
    */
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

    /* =======================================
	VALIDATION
       =======================================
    */
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
    }
  }//end if validate

} //end main
