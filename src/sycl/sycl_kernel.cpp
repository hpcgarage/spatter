#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <vector>
#include <iostream>
//#include "sycl_backend.h"
#include "sycl_dev_profile.hpp"
#include "sgtype.h"

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

            // Kernel
            auto kernel = [=]()
            {
                int idx_shared[MAX_IDX_LEN];
                int ngatherperblock = block[0] / idx_len; 

                for (int bid = 0; bid < grid[0]; ++bid)
                {
                    for (int tid = 0; tid < block[0]; ++tid)
                    {
                        if (tid < idx_len)
                            idx_shared[tid] = idxAccessor[tid];

                        int gatherid = tid / idx_len;
                        int src_offset =  (bid * ngatherperblock + gatherid) * delta;
                        double x;

                        x = srcAccessor[idx_shared[tid % idx_len] + src_offset]; 
                    }
                }
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

            // Kernel
            auto kernel = [=]()
            {
                int idx_shared[MAX_IDX_LEN];
                int ngatherperblock = block[0] / idx_len; 

                for (int bid = 0; bid < grid[0]; ++bid)
                {
                    for (int tid = 0; tid < block[0]; ++tid)
                    {
                        if (tid < idx_len)
                            idx_shared[tid] = idxAccessor[tid];

                        int gatherid = tid / idx_len;
                        int src_offset =  (bid * ngatherperblock + gatherid) * delta;
			int idx_shared_val = idx_shared[tid % idx_len];

                        srcAccessor[idx_shared_val + src_offset] = idx_shared_val; 
                    }
                }
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
