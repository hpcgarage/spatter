#include <CL/sycl.hpp>
#include <iostream>

int main() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        for (const auto& device : platform.get_devices()) {
            std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
            std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
            std::cout << "  Type: "
                      << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Other") << "\n";
            std::cout << "  Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
            std::cout << "  Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) << " MB\n";
            std::cout << "  Max work-group size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
            std::cout << "  Max work-item dimensions: ";
            auto dims = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            std::cout << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n\n";
        }
    }
    return 0;
}