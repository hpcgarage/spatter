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
#include "util.h"

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

#if defined ( USE_PAPI )
    #include "papi-util.h"
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
size_t seed;
size_t wrap = 1;
size_t R = 10;
size_t workers = 1;
size_t vector_len = 1;
size_t local_work_size = 1;
size_t ms1_gap = 0; 
size_t ms1_run = 0;

unsigned int shmem = 0;
int random_flag = 0;
int ms1_flag = 0;
int config_flag = 0;

int json_flag = 0, validate_flag = 0, print_header_flag = 1;

void print_header(){
    printf("backend kernel op time source_size target_size idx_size bytes_moved usable_bandwidth actual_bandwidth omp_threads vector_len block_dim\n");
}

void make_upper (char* s) {
    while (*s) {
        *s = toupper(*s);
        s++;
    }
}

long posmod (long i, long n) {
    return (i % n + n) % n;
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
    if(backend == OPENMP) printf("OPENMP ");
    if(backend == OPENCL) printf("OPENCL ");
    if(backend == CUDA) printf("CUDA ");

    if(kernel == SCATTER) printf("SCATTER ");
    if(kernel == GATHER) printf("GATHER ");
    if(kernel == SG) printf("SG ");

    if(op == OP_COPY) printf("COPY ");
    if(op == OP_ACCUM) printf("ACCUM ");

    printf("%lf %zu %zu %zu ", time, source_size, target_size, index_size);

    size_t bytes_moved = 2 * index_len * sizeof(sgData_t);
    double usable_bandwidth = bytes_moved / time / 1000. / 1000.;
    double actual_bandwidth = (bytes_moved + index_size) / time / 1000. / 1000.;
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

unsigned long ul_omp_get_thread_num()
{
  return (unsigned long) omp_get_thread_num();
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

    //Initialize PAPI with multi-threaded support
    #ifdef USE_PAPI
	if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
  	   exit(1);
	if (PAPI_thread_init(ul_omp_get_thread_num) != PAPI_OK)
  	   handle_error(1);

	//Open a file pointer to write PAPI results to
	FILE *papiFile, *papiConfig;
	papiFile = fopen("papi_output.txt", "w");
	//The config file contains 1) the number of events
	papiConfig = fopen("papi_config.txt", "r");
	if (papiFile == NULL)
	{
 	   perror("Opening the PAPI output file failed.");
    	   return 1;
	}
	else if (papiConfig == NULL)
	{
 	   perror("Opening the PAPI config file failed.");
    	   return 1;
	}

	//Set the event struct for PAPI
        #define max_events 4
	struct papi_t papi;
	int num_events = 0;
  	int events[max_events];
  	long long int counters[max_events];

	//Read in the config file
	fscanf(papiConfig,"%d", &num_events);
	papi.num = num_events;
        printf("%d events\n",papi.num);

	if(papi.num > max_events)
	{
		perror("Only up to 4 events supported!\n");
		return 1;
	}

	for(int i = 0; i < papi.num; i++)
	{
		// reads text until newline 
		fscanf(papiConfig,"%s", papi.event_names[i]);
		//printf("%d",i);
		fscanf(papiConfig,"%x", &(papi.events[i]));
		printf("%s: %x\n",papi.event_names[i],papi.events[i]);
		
		papi.counters[i] = 0;
	}
	
	//papi_struct_set(&papi, num_events, events, counters);
    #endif

    source.len = source_len;
    target.len = target_len;
    si.len     = index_len;
    ti.len     = index_len;

    /* These are the total size of the data allocated for each buffer */
    source.size = source.len * sizeof(sgData_t);
    target.size = target.len * sizeof(sgData_t);

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


    /*
#ifdef USE_CUDA
    si.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(si.size); 
    ti.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(ti.size); 
#elif defined USE_OPENMP
    si.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(si.size); 
    ti.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(ti.size); 
#elif defined USE_OPENCL
    si.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(si.size); 
    ti.host_ptr = (sgIdx_t*) sg_safe_cpu_alloc(ti.size); 
#endif
*/
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
    if (print_header_flag) print_header();
    

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
	    
	    #ifdef USE_PAPI
	    PAPI_start_counters(papi.events, papi.num);
	    #endif

            switch (kernel) {
                case SG:
                    if (op == OP_COPY) 
                        sg_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				(sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				index_len);
                    else 
                        sg_accum_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				      (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				      index_len);
                    break;
                case SCATTER:
                    if (op == OP_COPY)
				scatter_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
					     (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
					     index_len);
                    else
                        scatter_accum_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				           (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				           index_len);
                    break;
                case GATHER:
                    if (op == OP_COPY)
		        gather_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				    (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				    index_len);
                    else
                        gather_accum_omp ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				          (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				          index_len);
                    break;
                default:
                    printf("Error: Unable to determine kernel\n");
                    break;
            }

            double time_ms = sg_get_time_ms();
	    if (i!=0) report_time(time_ms/1000., source.size, target.size, si.size, vector_len);
            
	    #ifdef USE_PAPI
	    	PAPI_read_counters(papi.counters, papi.num);
	    	dump_papi_to_file(&papi,papiFile);
	    #endif
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
                        sg_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
				   (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
				   index_len);
                    else 
                        sg_accum_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
					 (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
					 index_len);
                    break;
                case SCATTER:
                    if (op == OP_COPY)
				scatter_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
						(sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
						index_len);
                    else
                        scatter_accum_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
					      (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
					      index_len);
                    break;
                case GATHER:
                    if (op == OP_COPY)
				gather_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
					       (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
					       index_len);
                    else
                        gather_accum_serial ((sgData_t * restrict)target.host_ptr, (long * restrict)ti.host_ptr,
					     (sgData_t * restrict)source.host_ptr, (long * restrict)si.host_ptr, 
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
      
    //Make sure PAPI is stopped and the file is closed
    #ifdef USE_PAPI
	   fclose(papiFile);
	   PAPI_shutdown();
   #endif

} //end main
