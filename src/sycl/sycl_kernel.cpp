#include <CL/sycl.hpp>
#include <chrono>
#include <vector>
#include "../../include/sgtype.h"
#include "sycl_dev_profile.hpp"

using namespace cl::sycl;

class Gather;

double sycl_gather(double* src, size_t src_len, sgIdx_t* idx, size_t idx_len, size_t delta, unsigned long global_work_size, unsigned long local_work_size)
{
    {
        // Enable profiling so that we can record the kernel execution time
        auto property_list = property_list{property::queue::enable_profiling()};

        // Device Selection (this will only run on GPUs or CPUs)
        default_selector device_selector;

        // Create the device queue
        queue device_queue(device_selector, NULL, property_list);

        // Create data structures for recording events
        std::vector<event> eventList;
        std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> startTimeList;

        // Create the buffers for accessing data
        buffer<double, 1> srcBuf(src, src_len * sizeof(double));
        buffer<double, 1> idxBuf(idx, idx_len * sizeof(sgIdx_t));

        // Define the dimensions of the operation
        range<2> numOfItems(global_work_size / local_work_size, local_work_size);

        // Record the start time
        startTimeList.push_back(std::chrono::high_resolution_clock::now());

        // Submit the kernel
        eventList.push_back(device_queue.submit([&](handler &cgh) 
        {
            // Create accessors
            auto srcAccessor = srcBuf.get_access<access::mode::read>(cgh);
            auto idxAccessor = idxBuf.get_access<access::mode::read>(cgh);

            // Kernel
            auto kernel = [=](id<2> id)
            {
                double x;

                // Ported over from CUDA kernel...
                int tid = id[1];
                int bid = id[0];
                int ngatherperblock = (global_work_size / local_work_size) / idx_len;
                int gatherid = tid / idx_len;

                int src_offset = (bid * ngatherperblock + gatherid) * delta;

                // Gather
                x = srcAccessor[idxAccessor[tid % idx_len] + offset];
            };

            cgh.parallel_for<Gather>(numOfItems, kernel);

        }));

        // Wait for asynchronous execution of kernel to complete
        device_queue.wait();

        // Get profile stats
        example_profiler<double> profiler(eventList, startTimeList);
        return profiler.get_kernel_execution_time();
    }
}