#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include <ctype.h>
#include "kernels/openmp_kernels.h"
#include "cl-helper.h"
#include "parse-args.h"
#include "sgtype.h"
#include "sgbuf.h"
#include "mytime.h"

#define alloc(size) aligned_alloc(64, size)

enum sg_backend backend = INVALID_BACKEND;
enum sg_kernel  kernel  = INVALID_KERNEL;

char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];

size_t source_len;
size_t target_len;
size_t index_len;
size_t block_len;
size_t seed;
size_t R = 10;
size_t N = 100;
size_t workers = 1;

int json_flag = 0, validate_flag = 0, print_header_flag = 1;

void print_header(){
    if (backend == OPENCL) printf("backend platform device kernel op time source_size target_size idx_size worksets bytes_moved usable_bandwidth loops runs workers\n");
    if (backend == OPENMP) printf("backend kernel op time source_size target_size idx_size worksets bytes_moved usable_bandwidth loops runs workers\n");
}

void make_upper (char* s) {
    while (*s) {
        *s = toupper(*s);
        s++;
    }
}

/** Time reported in seconds, sizes reported in bytes, bandwidth reported in mib/s"
 */
void report_time(double time, size_t source_size, size_t target_size, size_t idx_size, size_t worksets){
    if(backend == OPENMP) printf("OPENMP ");
    if(backend == OPENCL) printf("OPENCL ");

    if(backend == OPENCL){
        make_upper(platform_string);
        make_upper(device_string);
        printf("%s %s ", platform_string, device_string);
    }

    if(kernel == SCATTER) printf("SCATTER ");
    if(kernel == GATHER) printf("GATHER ");
    if(kernel == SG) printf("SG ");

    printf(SGOPSTRING);

    printf(" %lf %zu %zu %zu ", time, source_size, target_size, idx_size);
    printf("%zu ", worksets);

    size_t bytes_moved = idx_size * block_len * sizeof(SGTYPE_C) / worksets * N;
    double usable_bandwidth = bytes_moved / time / 1024. / 1024.;
    printf("%zu %lf ", bytes_moved, usable_bandwidth);
    printf("%zu %zu %zu", N, R, workers);

    printf("\n");

}

int main(int argc, char **argv)
{
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_mem_flags flags; 
    cl_kernel sgp;

    sgDataBuf  source;
    sgDataBuf  target;
    sgIndexBuf si; //source_index
    sgIndexBuf ti; //target_index

    size_t cpu_cache_size = 30720 * 1000; 
    size_t cpu_flush_size = cpu_cache_size * 8;
    cl_ulong device_cache_size = 0;
    size_t device_flush_size = 0;
    size_t worksets = 1;
    cl_uint work_dim = 1;
    size_t global_work_size = 1;
    size_t local_work_size = 1;
    cl_event e;
    
    char *kernel_string;

    /* Parse command line arguments */
    parse_args(argc, argv);


    /* Create a context and corresponding queue */
    if (backend == OPENCL) {
        create_context_on(platform_string, device_string, 0, 
                      &context, &queue, &device, 1);
    }

    source.len = source_len;
    target.len = target_len;
    si.len     = index_len;
    ti.len     = index_len;

    /* Determine how many worksets we will need to flush the cache */
    if (backend == OPENMP) {
        worksets = cpu_flush_size / 
            ((source.len + target.len) * sizeof(SGTYPE_C) 
            + (si.len + ti.len) * sizeof(cl_ulong)) + 1;
    }
    else if (backend == OPENCL) {
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, 
                sizeof(device_cache_size), &device_cache_size, NULL); 
        device_flush_size = device_cache_size * 8;
        worksets = device_flush_size / 
            ((source.len + target.len) * sizeof(SGTYPE_C) 
            + (si.len + ti.len) * sizeof(cl_ulong)) + 1;
    }

    /* These are the total size of the data allocated for each buffer */
    source.size = worksets * block_len * source.len * sizeof(SGTYPE_C);
    target.size = worksets * block_len * target.len * sizeof(SGTYPE_C);
    si.size     = worksets * si.len * sizeof(cl_ulong);
    ti.size     = worksets * ti.len * sizeof(cl_ulong);

    /* This is the number of SGTYPEs in a workset */
    source.block_len = block_len * source.len;
    target.block_len = block_len * target.len;

    /* Create the kernel */
    if (backend == OPENCL) {
        kernel_string = read_file(kernel_file);
        sgp = kernel_from_string(context, kernel_string, kernel_name, NULL);
        free(kernel_string);
    }

    /* Create buffers on host */
    source.host_ptr = (SGTYPE_C*) alloc(source.size); 
    target.host_ptr = (SGTYPE_C*) alloc(target.size); 
    si.host_ptr = (cl_ulong*) alloc(si.size); 
    ti.host_ptr = (cl_ulong*) alloc(ti.size); 

    /* Populate buffers on host */
    random_data(source.host_ptr, source.len * worksets);
    linear_indices(si.host_ptr, si.len, worksets);
    linear_indices(ti.host_ptr, ti.len, worksets);

    /* Create buffers on device and transfer data from host */
    if (backend == OPENCL) {

        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
        source.dev_ptr = clCreateBufferSafe(context, flags, source.size, source.host_ptr);
        si.dev_ptr = clCreateBufferSafe(context, flags, si.size, si.host_ptr);
        ti.dev_ptr = clCreateBufferSafe(context, flags, ti.size, ti.host_ptr);

        flags = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
        target.dev_ptr = clCreateBufferSafe(context, flags, target.size, NULL); 

        /* Time OpenCL Kernel */
        
        SET_10_KERNEL_ARGS(sgp, target.dev_ptr, ti.dev_ptr, source.dev_ptr, 
                si.dev_ptr, target.block_len, source.block_len, 
                index_len, worksets, N, block_len);

    }

    /* Begin benchmark */
    if (print_header_flag) print_header();
    if (backend == OPENCL) {

        for (int i = 0; i <= R; i++) {
            
            CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, sgp, work_dim, NULL, 
                       &global_work_size, &local_work_size, 
                      0, NULL, &e)); 
            clWaitForEvents(1, &e);

            cl_ulong start = 0, end = 0;
            size_t retsize;
            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,&retsize));
            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,&retsize));

            cl_ulong time_ns = end - start;
            double time_s = time_ns / 1000000000.;
            if (i!=0) report_time(time_s, source.size, target.size, si.size, worksets);

        }

    }

    /* Time OpenMP Kernel */

    if (backend == OPENMP) {

        omp_set_num_threads(workers);
        for (int i = 0; i <= R; i++) {
            zero_time();
            switch (kernel) {
                case SG:
                    sg_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        target.block_len, source.block_len, index_len, worksets, N, 
                        block_len);
                    break;
                case SCATTER:
                    scatter_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        target.block_len, source.block_len, index_len, worksets, N, 
                        block_len);
                    break;
                case GATHER:
                    gather_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        target.block_len, source.block_len, index_len, worksets, N, 
                        block_len);
                    break;
                default:
                    printf("Error: Unable to determine kernel\n");
                    break;
            }
            double time_ms = get_time();
            if (i!=0) report_time(time_ms/1000., source.size, target.size, si.size, worksets);

        }

    }
    /* Validate results  -- OPENMP assumed correct*/
    if(validate_flag && backend == OPENCL) {

        clEnqueueReadBuffer(queue, target.dev_ptr, 1, 0, target.size, 
                target.host_ptr, 0, NULL, &e);
        clWaitForEvents(1, &e);

        SGTYPE_C *target_backup_host = (SGTYPE_C*) alloc(target.size); 
        memcpy(target_backup_host, target.host_ptr, target.size);

        switch (kernel) {
            case SG:
                sg_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                    target.block_len, source.block_len, index_len, worksets, R, 
                    block_len);
                break;
            case SCATTER:
                scatter_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                    target.block_len, source.block_len, index_len, worksets, R, 
                    block_len);
                break;
            case GATHER:
                gather_omp(target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                    target.block_len, source.block_len, index_len, worksets, R, 
                    block_len);
                break;
        }


        for (size_t i = 0; i < target.len * worksets; i++) {
            if (target.host_ptr[i] != target_backup_host[i]) {
                printf(":(\n");
            }
        }
    }
}
