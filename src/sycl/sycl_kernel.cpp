#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <vector>
#include <iostream>
//#include "sycl_backend.h"
#include "sycl_dev_profile.hpp"
#include "sgtype.h"

#define USE_OPTIMIZED_SYCL

//For debugging on Intel FPGAs, we follow the guidance from this URL: https://community.intel.com/t5/Intel-oneAPI-Data-Parallel-C/calling-printf-from-within-kernel/td-p/1183830
#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif

using namespace cl::sycl;

#define MAX_IDX_LEN 2048

class Gather;
class Scatter;

extern "C" double sycl_gather(double* src, size_t src_size, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned int* grid, unsigned int* block, unsigned int dim)
{
    if (dim != 1)
    {
        std::cerr << "Error: dim != 1, unsupported dim size!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (idx_len > MAX_IDX_LEN)
    {
	std::cerr << "Error: idx_len exceeds MAX_IDX_LEN!" << std::endl;
	exit(EXIT_FAILURE);
    }

    {
        // Enable profiling so that we can record the kernel execution time
        property_list pl{property::queue::enable_profiling()};

        // Device Selection
        #if defined(CPU_HOST)
            std::cout << "Using CPU Host." << std::endl;
            host_selector device_selector;
        #elif defined(FPGA_EMULATOR)
            std::cout << "Using FPGA emulation." << std::endl;
            intel::fpga_emulator_selector device_selector;
        #else
            std::cout << "Using FPGA hardware (bitstream)." << std::endl;
            intel::fpga_selector device_selector;
        #endif

        // Create the device queue
        queue device_queue(device_selector, NULL, pl);

        // Create data structures for recording events
        std::vector<event> eventList;
        std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> startTimeList;

        //Query platform and device
        platform platform = device_queue.get_context().get_platform();
        sycl::device device = device_queue.get_device();
        std::cout << "Platform name: " <<  platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " <<  device.get_info<sycl::info::device::name>().c_str() << std::endl;

        // Create the buffers for accessing data
        buffer<double, 1> srcBuf(src, src_size);
        buffer<sgIdx_t, 1> idxBuf(idx, idx_len * sizeof(sgIdx_t));
        buffer<unsigned int, 1> gridBuf(grid, 1 * sizeof(unsigned int));
        buffer<unsigned int, 1> blockBuf(block, 1 * sizeof(unsigned int));
        //buffer<size_t, 1> idxLenBuf(&idx_len, 1 * sizeof(size_t));

        // Define the dimensions of the operation
        //range<1> numOfItems((global_work_size / local_work_size) * local_work_size);

        // Record the start time
        startTimeList.push_back(std::chrono::high_resolution_clock::now());

        // Submit the kernel
        eventList.push_back(device_queue.submit([&](handler &cgh) 
        {
            // Create accessors
            auto srcAccessor = srcBuf.get_access<access::mode::read>(cgh);
            auto idxAccessor = idxBuf.get_access<access::mode::read>(cgh);
            auto gridAccessor = gridBuf.get_access<access::mode::read>(cgh);
            auto blockAccessor = blockBuf.get_access<access::mode::read>(cgh);
            //auto idxLenAccessor = idxLenBuf.get_access<access::mode::read>(cgh);

            // Kernel
            auto kernel = [=]()
            {
                #ifdef USE_OPTIMIZED_SYCL

		//Adding printf "breakpoints" to the kernel 
                static const CONSTANT char FMT[] = "n: %u\n";
                sycl::intel::experimental::printf(FMT, 1);

                // Create local vars when possible
                size_t idx_len_local = idx_len;
                size_t delta_local = delta;
                int idx_shared[MAX_IDX_LEN];
                int ngatherperblock = blockAccessor[0] / idx_len_local;
                int gridDim = gridAccessor[0];
                int blockDim = blockAccessor[0];
                
		sycl::intel::experimental::printf(FMT, 2);
	
                // First, perform setting of idx_shared
                // Unroll this because it can and should be done in parallel
                #pragma unroll
                for (int i = 0; i < idx_len_local; ++i)
                    idx_shared[i] = idxAccessor[i];

		sycl::intel::experimental::printf(FMT, 3);

                // Next, condense nested loop into a single loop
                // Unroll this loop as well
                #pragma unroll
                for (int i = 0; i < gridDim * blockDim; ++i)
                {
                    // Create local vars as deep as possible
                    int tid = i % blockDim;
                    int bid = i / blockDim;
                    int gatherid = tid / idx_len_local;
                    int src_offset = (bid * ngatherperblock * gatherid) * delta_local;
                    int idx_shared_val = idx_shared[gatherid];
                    int src_index = idx_shared_val + src_offset;
                    double x;

                    x = srcAccessor[src_index];
                }
                
		sycl::intel::experimental::printf(FMT, 4);

		//Non-optimized version of this SYCL kernel from the OneAPI guide
                #else
		
                int idx_shared[MAX_IDX_LEN];
                //int ngatherperblock = block[0] / idx_len; 
                //int ngatherperblock = (int) ((size_t) block[0]) / idx_len;
                //int ngatherperblock = blockAccessor[0] / idxLenAccessor[0];
                int ngatherperblock = blockAccessor[0] / idx_len;
                for (int bid = 0; bid < gridAccessor[0]; ++bid)
                {
                    for (int tid = 0; tid < blockAccessor[0]; ++tid)
                    {
                        if (tid < idx_len)
                            idx_shared[tid] = idxAccessor[tid];

                        int gatherid = tid / idx_len;
                        int src_offset =  (bid * ngatherperblock + gatherid) * delta;
                        double x;

                        x = srcAccessor[idx_shared[tid % idx_len] + src_offset]; 
                    }
                }
		
                //double x = 1.0 + 2.0;
                #endif
            };

            cgh.single_task<Gather>(kernel);
        }));

        // Wait for asynchronous execution of kernel to complete
        device_queue.wait();

        // Get profile stats
        example_profiler<double> profiler(eventList, startTimeList);
        return profiler.get_kernel_execution_time();
    }
}

extern "C" double sycl_scatter(double* src, size_t src_size, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned int* grid, unsigned int* block, unsigned int dim)
{
    if (dim != 1)
    {
        std::cerr << "Error: dim != 1, unsupported dim size!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (idx_len > MAX_IDX_LEN)
    {
        std::cerr << "Error: idx_len exceeds MAX_IDX_LEN!" << std::endl;
        exit(EXIT_FAILURE);
    }

    {
        // Enable profiling so that we can record the kernel execution time
        property_list pl{property::queue::enable_profiling()};

        // Device Selection
        #if defined(CPU_HOST)
            host_selector device_selector;
        #elif defined(FPGA_EMULATOR)
            intel::fpga_emulator_selector device_selector;
        #else
            intel::fpga_selector device_selector;
        #endif

        // Create the device queue
        queue device_queue(device_selector, NULL, pl);

        // Create data structures for recording events
        std::vector<event> eventList;
        std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> startTimeList;

        //Query platform and device
        platform platform = device_queue.get_context().get_platform();
        sycl::device device = device_queue.get_device();
        std::cout << "Platform name: " <<  platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " <<  device.get_info<sycl::info::device::name>().c_str() << std::endl;

        // Create the buffers for accessing data
        buffer<double, 1> srcBuf(src, src_size);
        buffer<sgIdx_t, 1> idxBuf(idx, idx_len * sizeof(sgIdx_t));
        buffer<unsigned int, 1> gridBuf(grid, 1 * sizeof(unsigned int));
        buffer<unsigned int, 1> blockBuf(block, 1 * sizeof(unsigned int));

        // Define the dimensions of the operation
        //range<1> numOfItems((global_work_size / local_work_size) * local_work_size);

        // Record the start time
        startTimeList.push_back(std::chrono::high_resolution_clock::now());

        // Submit the kernel
        eventList.push_back(device_queue.submit([&](handler &cgh) 
        {
            // Create accessors
            auto srcAccessor = srcBuf.get_access<access::mode::write>(cgh);
            auto idxAccessor = idxBuf.get_access<access::mode::read>(cgh);
            auto gridAccessor = gridBuf.get_access<access::mode::read>(cgh);
            auto blockAccessor = blockBuf.get_access<access::mode::read>(cgh);

            // Kernel
            auto kernel = [=]()
            {
                #ifdef USE_OPTIMIZED_SYCL

                // Create local vars when possible
                size_t idx_len_local = idx_len;
                size_t delta_local = delta;
                int idx_shared[MAX_IDX_LEN];
                int ngatherperblock = blockAccessor[0] / idx_len_local;
                int gridDim = gridAccessor[0];
                int blockDim = blockAccessor[0];

                // First, perform setting of idx_shared
                // Unroll this because it can and should be done in parallel
                #pragma unroll
                for (int i = 0; i < idx_len_local; ++i)
                    idx_shared[i] = idxAccessor[i];

                // Next, condense nested loop into a single loop
                // Unroll this loop as well
                #pragma unroll
                for (int i = 0; i < gridDim * blockDim; ++i)
                {
                    // Create local vars as deep as possible
                    int tid = i % blockDim;
                    int bid = i / blockDim;
                    int gatherid = tid / idx_len_local;
                    int src_offset = (bid * ngatherperblock * gatherid) * delta_local;
                    int idx_shared_val = idx_shared[gatherid];
                    int src_index = idx_shared_val + src_offset;

                    srcAccessor[src_index] = idx_shared_val;
                }

                #else
                int idx_shared[MAX_IDX_LEN];
                //int ngatherperblock = block[0] / idx_len; 
                int ngatherperblock = blockAccessor[0] / idx_len;
                
                for (int bid = 0; bid < gridAccessor[0]; ++bid)
                {
                    for (int tid = 0; tid < blockAccessor[0]; ++tid)
                    {
                        if (tid < idx_len)
                            idx_shared[tid] = idxAccessor[tid];

                        int gatherid = tid / idx_len;
                        int src_offset =  (bid * ngatherperblock + gatherid) * delta;
			            int idx_shared_val = idx_shared[tid % idx_len];

                        srcAccessor[idx_shared_val + src_offset] = idx_shared_val; 
                    }
                }
                #endif
            };

            cgh.single_task<Scatter>(kernel);
        }));

        // Wait for asynchronous execution of kernel to complete
        device_queue.wait();

        // Get profile stats
        example_profiler<double> profiler(eventList, startTimeList);
        return profiler.get_kernel_execution_time();
    }
}
