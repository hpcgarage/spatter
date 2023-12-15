/*!
  \file Configuration.hh
*/

#ifndef SPATTER_CONFIGURATION_HH
#define SPATTER_CONFIGURATION_HH

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "CudaBackend.hh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cctype>
#include <experimental/iterator>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include "Spatter/SpatterTypes.hh"
#include "Spatter/Timer.hh"

namespace Spatter {

class ConfigurationBase {
public:
  ConfigurationBase(std::string k, const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t wrap, const size_t count, const int nthreads,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : kernel(k), pattern(pattern), pattern_gather(pattern_gather),
        pattern_scatter(pattern_scatter), delta(delta), wrap(wrap),
        count(count), omp_threads(nthreads), nruns(nruns), verbosity(verbosity),
        time_seconds(0) {
    std::transform(kernel.begin(), kernel.end(), kernel.begin(),
        [](unsigned char c) { return std::tolower(c); });
  }

  ~ConfigurationBase() = default;

  virtual int run(bool timed) {
    if (kernel.compare("gather") == 0)
      gather(timed);
    else if (kernel.compare("scatter") == 0)
      scatter(timed);
    else if (kernel.compare("sg") == 0)
      scatter_gather(timed);
    else if (kernel.compare("multigather") == 0)
      multi_gather(timed);
    else if (kernel.compare("multiscatter") == 0)
      multi_scatter(timed);
    else {
      std::cerr << "Invalid Kernel Type" << std::endl;
      return -1;
    }

    return 0;
  }

  virtual void gather(bool timed) = 0;
  virtual void scatter(bool timed) = 0;
  virtual void scatter_gather(bool timed) = 0;
  virtual void multi_gather(bool timed) = 0;
  virtual void multi_scatter(bool timed) = 0;

  virtual void report() {
    std::cout << nruns * pattern.size() * sizeof(size_t) << " Total Bytes Moved"
              << std::endl;
    std::cout << pattern.size() * sizeof(size_t) << " Bytes Moved per Run"
              << std::endl;
    std::cout << nruns << " Runs took " << std::fixed << time_seconds
              << " Seconds" << std::endl;
    std::cout << "Average Bandwidth: "
              << (double)(nruns * pattern.size() * sizeof(size_t)) /
            time_seconds / 1000000.
              << " MB/s" << std::endl;
  }

  virtual void setup() {
    if (kernel.compare("multigather")) {
      if (pattern.size() == 0) {
        std::cerr << "Pattern needs to have length of at least 1" << std::endl;
        exit(1);
      }
      if (pattern_gather.size() == 0) {
        std::cerr << "Pattern-Gather needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
    } else if (kernel.compare("multiscatter")) {
      if (pattern.size() == 0) {
        std::cerr << "Pattern needs to have length of at least 1" << std::endl;
        exit(1);
      }
      if (pattern_scatter.size() == 0) {
        std::cerr << "Pattern-Scatter needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
    } else if (kernel.compare("sg")) {
      if (pattern_gather.size() == 0) {
        std::cerr << "Pattern-Gather needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
      if (pattern_scatter.size() == 0) {
        std::cerr << "Pattern-Scatter needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
    } else {
      if (pattern.size() == 0) {
        std::cerr << "Pattern needs to have length of at least 1" << std::endl;
        exit(1);
      }
    }

    dense.resize(pattern.size());

    for (size_t i = 0; i < pattern.size(); ++i)
      dense[i] = rand();

    size_t max_pattern_val = 0;
    for (size_t i = 0; i < pattern.size(); ++i)
      if (pattern[i] > max_pattern_val)
        max_pattern_val = pattern[i];

    sparse.resize(max_pattern_val + 1);

    for (size_t i = 0; i < max_pattern_val + 1; ++i)
      sparse[i] = rand();

    if (verbosity >= 3)
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDense Array Size: " << dense.size()
                << "\tSparse Array Size: " << sparse.size()
                << "\tMax Pattern Val: " << max_pattern_val << std::endl;
  }

public:
  const std::string name;

  std::string kernel;
  const std::vector<size_t> pattern;
  const std::vector<size_t> pattern_gather;
  const std::vector<size_t> pattern_scatter;

  std::vector<double> sparse;
  std::vector<double> sparse_gather;
  std::vector<double> sparse_scatter;

  std::vector<double> dense;

  const size_t delta;
  const std::vector<size_t> deltas;
  const size_t wrap;
  const size_t count;

  int seed;
  size_t vector_len;
  size_t shmem;
  size_t local_work_size;
  size_t op;

  int ro_morton;
  int ro_hilbert;
  int ro_block;

  int stride_kernel;

  const int omp_threads;
  const unsigned long nruns;
  const unsigned long verbosity;

  Spatter::Timer timer;
  double time_seconds;
};

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  if (config.verbosity >= 1)
    config_output << "Kernel: " << config.kernel;

  if (config.verbosity >= 2) {
    config_output << "\nPattern: ";
    std::copy(std::begin(config.pattern), std::end(config.pattern),
        std::experimental::make_ostream_joiner(config_output, ", "));
  }
  return out << config_output.str() << std::endl;
}

template <typename Backend> class Configuration : public ConfigurationBase {};

template <> class Configuration<Spatter::Serial> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t wrap, const size_t count, const unsigned long nruns = 10,
      const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, pattern_gather, pattern_scatter,
            delta, wrap, count, 1, nruns, verbosity) {
    setup();
  };

  void gather(bool timed) {
    size_t pattern_length = pattern.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern_length; ++i)
      dense[i] = sparse[pattern[i]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void scatter(bool timed) {
    size_t pattern_length = pattern.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern_length; ++i)
      sparse[pattern[i]] = dense[i];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void scatter_gather(bool timed) {
    assert(pattern_scatter.size() == pattern_gather.size());
    size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern_length; ++i)
      sparse_scatter[pattern_scatter[i]] = sparse_gather[pattern_gather[i]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_gather(bool timed) {
    size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern_length; ++i)
      dense[i] = sparse[pattern[pattern_gather[i]]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_scatter(bool timed) {
    size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern_length; ++i)
      sparse[pattern[pattern_scatter[i]]] = dense[i];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void report() {
    std::cout << "Spatter Serial Report" << std::endl;

    ConfigurationBase::report();
  }

  void setup() {
    if (verbosity >= 3)
      std::cout << "Spatter Serial Setup" << std::endl;

    ConfigurationBase::setup();
  }
};

#ifdef USE_OPENMP
template <> class Configuration<Spatter::OpenMP> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t wrap, const size_t count, const int nthreads,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, pattern_gather, pattern_scatter,
            delta, wrap, count, nthreads, nruns, verbosity) {
    setup();
  };

  int run(bool timed) {
    omp_set_num_threads(omp_threads);
    return ConfigurationBase::run(timed);
  }

  void gather(bool timed) {
    size_t pattern_length = pattern.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

#pragma omp parallel for simd
    for (size_t i = 0; i < pattern_length; ++i)
      dense[i] = sparse[pattern[i]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void scatter(bool timed) {
    size_t pattern_length = pattern.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

#pragma omp parallel for simd
    for (size_t i = 0; i < pattern_length; ++i)
      sparse[pattern[i]] = dense[i];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void scatter_gather(bool timed) {
    assert(pattern_scatter.size() == pattern_gather.size());
    size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

#pragma omp parallel for simd
    for (size_t i = 0; i < pattern_length; ++i)
      sparse_scatter[pattern_scatter[i]] = sparse_gather[pattern_gather[i]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_gather(bool timed) {
    size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

#pragma omp parallel for simd
    for (size_t i = 0; i < pattern_length; ++i)
      dense[i] = sparse[pattern[pattern_gather[i]]];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_scatter(bool timed) {
    size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

#pragma omp parallel for simd
    for (size_t i = 0; i < pattern_length; ++i)
      sparse[pattern[pattern_scatter[i]]] = dense[i];

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void report() {
    std::cout << "Spatter OpenMP Report" << std::endl;

    ConfigurationBase::report();
  }

  void setup() {
    if (verbosity >= 3)
      std::cout << "Spatter OpenMP Setup" << std::endl;

    ConfigurationBase::setup();
  }
};
#endif

#ifdef USE_CUDA
template <> class Configuration<Spatter::CUDA> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, conat size_t delta,
      const size_t wrap, const size_t count, const unsigned long nruns = 10,
      const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, pattern_gather, pattern_scatter,
            delta, wrap, count, 1, nruns, verbosity) {
    setup();
  };

  ~Configuration() {
    std::cout << "Deleting Configuration" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_pattern);
    cudaFree(dev_sparse);
    cudaFree(dev_dense);
  }

  int run(bool timed) {
    ConfigurationBase::run(timed);

    if (verbosity >= 3)
      std::cout << "Copying Vectors back to CPU" << std::endl;

    cudaMemcpy(sparse.data(), dev_sparse, sizeof(double) * sparse.size(),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(dense.data(), dev_dense, sizeof(double) * dense.size(),
        cudaMemcpyDeviceToHost);

    if (verbosity >= 3)
      std::cout << "Synchronizing CUDA Device" << std::endl;

    cudaDeviceSynchronize();

    if (verbosity >= 3) {
      std::cout << "Pattern: ";
      for (size_t val : pattern)
        std::cout << val << " ";
      std::cout << std::endl;
      std::cout << "Sparse: ";
      for (double val : sparse)
        std::cout << val << " ";
      std::cout << std::endl;
      std::cout << "Dense: ";
      for (double val : dense)
        std::cout << val << " ";
      std::cout << std::endl;
    }

    return 0;
  }

  void gather(bool timed) {
    int pattern_length = static_cast<int>(pattern.size());
    cudaDeviceSynchronize();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      cudaEventRecord(start);

    cuda_gather_wrapper(dev_pattern, dev_sparse, dev_dense, pattern_length);

    if (timed) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
    } else
      cudaDeviceSynchronize();

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_seconds += ((double)time_ms / 1000.0);
  }

  void scatter(bool timed) {
    int pattern_length = static_cast<int>(pattern.size());
    cudaDeviceSynchronize();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      cudaEventRecord(start);

    cuda_scatter_wrapper(dev_pattern, dev_sparse, dev_dense, pattern_length);

    if (timed) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
    } else
      cudaDeviceSynchronize();

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_seconds += ((double)time_ms / 1000.0);
  }

  void scatter_gather(bool timed) {
    assert(pattern_scatter.size() == pattern_gather.size());
    int pattern_length = static_cast<int> pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_scatter_gather_wrapper(dev_pattern_scatter, dev_sparse_scatter,
        dev_pattern_gather, dev_sparse_gather, pattern_length);

    /*
        for (size_t i = 0; i < pattern_length; ++i)
          sparse_scatter[pattern_scatter[i]] = sparse_gather[pattern_gather[i]];
    */

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_gather(bool timed) {
    int pattern_length = static_cast<int> pattern_gather.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_multi_gather_wrapper(
        dev_pattern, dev_pattern_gather, dev_sparse, dev_dense, pattern_length);

    /*
        for (size_t i = 0; i < pattern_length; ++i)
          dense[i] = sparse[pattern[pattern_gather[i]]];
    */

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_scatter(bool timed) {
    int pattern_length = static_cast<int> pattern_scatter.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter, dev_sparse,
        dev_dense, pattern_length);

    /*
        for (size_t i = 0; i < pattern_length; ++i)
          sparse[pattern[pattern_scatter[i]]] = dense[i];
    */

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void report() {
    std::cout << "Spatter CUDA Report" << std::endl;

    ConfigurationBase::report();
  }

  void setup() {
    if (verbosity >= 1) {
      std::cout << "Spatter CUDA Setup" << std::endl;

      int num_devices = 0;
      cudaGetDeviceCount(&num_devices);

      int gpu_id = 0;

      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, gpu_id);

      std::cout << "Device Number: " << gpu_id << std::endl;
      std::cout << "\tDevice Name: " << prop.name << std::endl;
      std::cout << "\tMemory Clock Rate (KHz): " << prop.memoryClockRate
                << std::endl;
      std::cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth
                << std::endl;
      std::cout << "\tPeak Memory Bandwidth (GB/s): "
                << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) /
              1.0e6
                << std::endl;
    }

    ConfigurationBase::setup();

    if (verbosity >= 3)
      std::cout << "Creating CUDA Events" << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    if (verbosity >= 3)
      std::cout << "Allocating Vectors on CUDA Device" << std::endl;

    cudaMalloc((void **)&dev_pattern, sizeof(size_t) * pattern.size());
    cudaMalloc((void **)&dev_sparse, sizeof(double) * sparse.size());
    cudaMalloc((void **)&dev_dense, sizeof(double) * dense.size());

    if (verbosity >= 3)
      std::cout << "Copying Vectors on to CUDA Device" << std::endl;

    cudaMemcpy(dev_pattern, pattern.data(), sizeof(size_t) * pattern.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sparse, sparse.data(), sizeof(double) * sparse.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dense, dense.data(), sizeof(double) * dense.size(),
        cudaMemcpyHostToDevice);

    if (verbosity >= 3)
      std::cout << "Synchronizing CUDA Device" << std::endl;

    cudaDeviceSynchronize();

    if (verbosity >= 3) {
      std::cout << "Pattern: ";
      for (size_t val : pattern)
        std::cout << val << " ";
      std::cout << std::endl;
      std::cout << "Sparse: ";
      for (double val : sparse)
        std::cout << val << " ";
      std::cout << std::endl;
      std::cout << "Dense: ";
      for (double val : dense)
        std::cout << val << " ";
      std::cout << std::endl;
    }
  }

public:
  size_t *dev_pattern;
  size_t *dev_pattern_gather;
  size_t *dev_pattern_scatter;

  double *dev_sparse;
  double *dev_sparse_gather;
  double *dev_sparse_scatter;

  double *dev_dense;

  cudaEvent_t start;
  cudaEvent_t stop;
};
#endif

} // namespace Spatter

#endif
