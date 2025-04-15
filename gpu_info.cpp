#include <CL/sycl.hpp>
#include <iostream>

int main() {
    sycl::gpu_selector selector;
    sycl::queue q(selector);
    auto device = q.get_device();

    std::cout << "Using SYCL device: " << device.get_info<sycl::info::device::name>() << "\n";
    std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "  Type: GPU\n";
    std::cout << "  Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "  Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) << " MB\n";
    std::cout << "  Max work-group size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
    auto dims = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    std::cout << "  Max work-item dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n\n";
    
    return 0;
}