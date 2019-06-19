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

//Strings defining program behavior
char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];

int validate_flag = 0, print_header_flag = 1;
int noaggregate_flag = 0;

//TODO: this shouldn't print out info about rc - only the system
void print_system_info(struct run_config rc){

    printf("\nRunning Spatter version 0.0\n");
    //printf("Contributors: Patrick Lavin, Jeff Young, Aaron Vose\n");
    printf("Backend: ");

    if(backend == OPENMP) printf("OPENMP\n");
    if(backend == OPENCL) printf("OPENCL\n");
    if(backend == CUDA) printf("CUDA\n");

    
    printf("Aggregate Results? %s", noaggregate_flag ? "NO" : "YES"); 
    
    //if (noaggregate_mode)


    printf("\n\n");
}

void print_header(){
    //printf("kernel op time source_size target_size idx_len bytes_moved actual_bandwidth omp_threads vector_len block_dim shmem\n");
    printf("%-7s %-12s %-12s\n", "config", "time(s)","bandwidth(MB/s)");
}

void *sg_safe_cpu_alloc (size_t size) {
    void *ptr = aligned_alloc (ALIGNMENT, size);
    if (!ptr) {
        printf("Falied to allocate memory on cpu: requested size %zu\n", size);
        exit(1);
    }
    return ptr;
}

/** Time reported in seconds, sizes reported in bytes, bandwidth reported in mib/s"
 */
void report_time(int ii, double time,  struct run_config rc){

    /*
    if(rc.kernel == SCATTER) printf("SCATTER ");
    if(rc.kernel == GATHER) printf("GATHER ");
    if(rc.kernel == SG) printf("SG ");

    if(rc.op == OP_COPY) printf("COPY ");
    if(rc.op == OP_ACCUM) printf("ACCUM ");

    printf("%lf %zu %zu ", time, source_size, target_size);

    printf("%zu ", rc.pattern_len);
    */

    size_t bytes_moved = 0;
    double usable_bandwidth = 0;
    double actual_bandwidth = 0;
    
    bytes_moved = 2 * sizeof(sgData_t) * rc.pattern_len * rc.generic_len;
    usable_bandwidth = 0;
    actual_bandwidth = bytes_moved / time / 1000. / 1000.;

    /*
    printf("%zu %lf ", bytes_moved, actual_bandwidth);

    printf("%zu %zu %zu %u", rc.omp_threads, rc.vector_len, rc.local_work_size, rc.shmem);

    printf("\n");
    */
    //printf("%d %.2e %.2e\n", 0, time, actual_bandwidth);
    printf("%-7d %-12.4g %-12.6g\n", ii, time, actual_bandwidth);

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
    size_t global_work_size = 1;
    char   *kernel_string;

    #ifdef USE_OPENCL
    cl_uint work_dim = 1;
    #endif
    
    // =======================================
    // Parse Command Line Arguments
    // =======================================
    
    struct run_config rc;
    rc = parse_args(argc, argv);

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
    
    if (rc.kernel == GATHER) {
        // the target only has rc.wrap slots of size rc.pattern_len to be gathered into. 
        target.size = rc.pattern_len * sizeof(sgData_t) * rc.wrap;
        target.len = target.size / sizeof(sgData_t);
        
        // we will duplicate the target space for every thread.
        target.nptrs = rc.omp_threads;

        // we must make sure there is sufficient space in source for us to slide the pattern
        size_t max_pattern_val = rc.pattern[0];
        for (size_t i = 0; i < rc.pattern_len; i++) {
            if (rc.pattern[i] > max_pattern_val) {
                max_pattern_val = rc.pattern[i];
            }
        }
                   
        source.size = ((max_pattern_val + 1) + (rc.generic_len-1)*rc.delta) * sizeof(double);
        source.len = source.size / sizeof(sgData_t);
    } else {
        //TODO: Add data allocation for SCATTER
        printf(" ERROR - only GATHER is currently supported\n");
        exit(1);
    }

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
    source.host_ptr = (sgData_t*) sg_safe_cpu_alloc(source.size); 

    // replicate the target space for every thread
    target.host_ptrs = (sgData_t**) sg_safe_cpu_alloc(target.nptrs * sizeof(sgData_t*));
    for (size_t i = 0; i < target.nptrs; i++) {
        target.host_ptrs[i] = (sgData_t*) sg_safe_cpu_alloc(target.size);
    }

    // Populate buffers cn host 
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
    if (backend == CUDA) {
        //TODO: Rewrite to not take index buffers
        //create_dev_buffers_cuda(&source, &target, &si, &ti);
        cudaMemcpy(source.dev_ptr_cuda, source.host_ptr, source.size, cudaMemcpyHostToDevice);
        /*
        cudaMemcpy(si.dev_ptr_cuda, si.host_ptr, si.size, cudaMemcpyHostToDevice);
        cudaMemcpy(ti.dev_ptr_cuda, ti.host_ptr, ti.size, cudaMemcpyHostToDevice);
        */
        cudaDeviceSynchronize();
    }
    #endif

    
    // =======================================
    // Execute Benchmark
    // =======================================
    
    int nrc=2;
    struct run_config *rc2 = (struct run_config *)malloc(sizeof(struct run_config) * 2);
    rc2[0] = rc;
    rc2[1] = rc;

    // Print some header info
    if (print_header_flag) 
    {
        print_system_info(rc);
        emit_configs(rc2, 2);
        print_header();
    }

    // Print config info

    

    for (int k = 0; k < nrc; k++) {
        // Time OpenCL Kernel 
        #ifdef USE_OPENCL
        if (backend == OPENCL) {

            //TODO: Rewrite without index buffers
            /*
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
                if (i!=0) report_time(time_s, source.size, target.size, si.size, vector_len, rc);

            }
            */

        }
        #endif // USE_OPENCL

        // Time CUDA Kernel 
        #ifdef USE_CUDA
        if (backend == CUDA) {

            //TODO: Rewrite without index buffers
            /*
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
                if (i!=0) report_time(time_s, source.size, target.size, si.size, vector_len, rc);

            }
            */

        }
        #endif // USE_CUDA



        // Time OpenMP Kernel 
        #ifdef USE_OPENMP
        if (backend == OPENMP) {
            omp_set_num_threads(rc2[k].omp_threads);
            double min_time_ms = 1e10;

            for (int i = 0; i <= rc2[k].nruns; i++) {

                sg_zero_time();

                switch (rc2[k].kernel) {
                    case SG:
                        if (rc2[k].op == OP_COPY) {
                            //sg_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr,index_len);
                        } else {
                            //sg_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case SCATTER:
                        if (rc2[k].op == OP_COPY) {
                            // scatter_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        } else {
                            // scatter_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case GATHER:
                        if (rc2[k].deltas_len <= 1) {
                            gather_smallbuf(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].delta, rc2[k].generic_len, rc2[k].wrap);
                        } else {
                            gather_smallbuf_multidelta(target.host_ptrs, source.host_ptr, rc2[k].pattern, rc2[k].pattern_len, rc2[k].deltas_ps, rc2[k].generic_len, rc2[k].wrap, rc2[k].deltas_len);
                        }
                        break;
                    default:
                        printf("Error: Unable to determine kernel\n");
                        break;
                }

                double time_ms = sg_get_time_ms();
                if (i!=0) {
                    if (noaggregate_flag) {
                        report_time(k, time_ms/1000., rc2[k]);
                    } else {
                        if (time_ms < min_time_ms) {
                            min_time_ms = time_ms;
                        }
                    }
                }
            }
            if (!noaggregate_flag) {
                report_time(k, min_time_ms/1000., rc2[k]);
            }
        }
        #endif // USE_OPENMP
        
        // Time Serial Kernel 
        #ifdef USE_SERIAL
        if (backend == SERIAL) {

            for (int i = 0; i <= R; i++) {

                sg_zero_time();

                //TODO: Rewrite serial kernel
                switch (rc2[k].kernel) {
                    case SG:
                        if (rc2[k].op == OP_COPY) {
                            //sg_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        } else {
                            //sg_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case SCATTER:
                        if (rc2[k].op == OP_COPY) {
                            //scatter_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        } else {
                            //scatter_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    case GATHER:
                        if (rc2[k].op == OP_COPY) {
                            //gather_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        } else {
                            //gather_accum_serial (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, index_len);
                        }
                        break;
                    default:
                        printf("Error: Unable to determine kernel\n");
                        break;
                }

                double time_ms = sg_get_time_ms();
                if (i!=0) report_time(k, time_ms/1000., rc2[k]);
            }
        }
        #endif // USE_SERIAL
    }
    

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

        // Pattern 
        printf("\'pattern\':[");
        for (int j = 0; j < rc[i].pattern_len; j++) {
            printf("%zu", rc[i].pattern[j]);
            if (j != rc[i].pattern_len-1) {
                printf(",");
            }
        }
        printf("], ");

        printf("\'delta\':%d}", rc[i].delta);

        if (i != nconfigs-1) {
            printf(",\n");
        }

    }
    printf(" ]\n\n");
}
