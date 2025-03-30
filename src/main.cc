#ifdef USE_MPI
#include "mpi.h"
#endif

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

#define xstr(s) str(s)
#define str(s) #s

void print_build_info(Spatter::ClArgs &cl) {
  std::cout << std::endl;
  std::cout << "Running Spatter version 1.1" << std::endl;
  std::cout << "Compiler: " << xstr(SPAT_CXX_NAME) << " ver. "
            << xstr(SPAT_CXX_VER) << std::endl;
  std::cout << "Backend: ";
  if (cl.backend.compare("serial") == 0)
    std::cout << "Serial" << std::endl;
  else if (cl.backend.compare("openmp") == 0)
    std::cout << "OpenMP" << std::endl;
  else if (cl.backend.compare("cuda") == 0)
    std::cout << "CUDA" << std::endl;
  else if (cl.backend.compare("oneapi") == 0)
    std::cout << "OneAPI" << std::endl;

  std::cout << "Aggregate Results? ";
  if (cl.aggregate == true)
    std::cout << "YES" << std::endl;
  else
    std::cout << "NO" << std::endl;

#ifdef USE_CUDA
  int gpu_id = 0;
  if (cl.backend.compare("cuda") == 0) {
    int num_devices = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));

    struct cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, gpu_id));

    std::cout << "Number of Devices: " << num_devices << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Memory Clock Rage (KHz): " << prop.memoryClockRate
              << std::endl;
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth
              << std::endl;
    std::cout << "Peak Memory Bandwidth (GB/s): "
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
              << std::endl;
  }
#endif

// #ifdef USE_ONEAPI
// for (const auto& device : platform.get_devices()) {
//   std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
//   std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
//   std::cout << "  Type: "
//             << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Other") << "\n";
//   std::cout << "  Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
//   std::cout << "  Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) << " MB\n";
//   std::cout << "  Max work-group size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
//   std::cout << "  Max work-item dimensions: ";
//   auto dims = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
//   std::cout << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n\n";
// }
// #endif

#ifdef USE_ONEAPI
  std::cout << "oneapi configuration (to be added)" << std::endl;
#endif

  std::cout << std::endl;
}

int main(int argc, char **argv) {

#ifdef USE_MPI
  MPI_Init(&argc, &argv);

  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  const unsigned long warmup_runs = 10;
  bool timed = 0;

  Spatter::ClArgs cl;
  if (Spatter::parse_input(argc, argv, cl) != 0)
    return -1;

#ifdef USE_MPI
  if (rank == 0) {
#endif
    if (cl.verbosity >= 1)
      print_build_info(cl);

    if (cl.verbosity >= 2)
      std::cout << cl;

    cl.report_header();
#ifdef USE_MPI
  }
#endif

  // for (std::unique_ptr<Spatter::ConfigurationBase> const &config : cl.configs) {
  for (std::shared_ptr<Spatter::ConfigurationBase> const &config : cl.configs) {
    for (unsigned long run = 0; run < (config->nruns + warmup_runs); ++run) {

      unsigned long run_id = 0;
      if (run >= warmup_runs) {
        timed = 1;
        run_id = run - warmup_runs;
      } else {
        timed = 0;
      }

      if (config->run(timed, run_id) != 0)
        return -1;
    }

#ifdef USE_MPI
    if (rank == 0) {
#endif
    config->report();
#ifdef USE_MPI
    }
#endif
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif
}
