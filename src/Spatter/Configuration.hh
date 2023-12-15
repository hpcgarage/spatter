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
  ConfigurationBase(const std::string name, std::string k,
      const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const size_t wrap,
      const size_t count, const int nthreads, const unsigned long nruns = 10,
      const bool aggregate = false, const bool compress = false,
      const unsigned long verbosity = 3)
      : name(name), kernel(k), pattern(pattern), pattern_gather(pattern_gather),
        pattern_scatter(pattern_scatter), delta(delta),
        delta_gather(delta_gather), delta_scatter(delta_scatter), wrap(wrap),
        count(count), omp_threads(nthreads), nruns(nruns), aggregate(aggregate),
        compress(compress), verbosity(verbosity), time_seconds(0) {
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
    if (kernel.compare("multigather") == 0) {
      if (pattern.size() == 0) {
        std::cerr << "Pattern needs to have length of at least 1" << std::endl;
        exit(1);
      }
      if (pattern_gather.size() == 0) {
        std::cerr << "Pattern-Gather needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
    } else if (kernel.compare("multiscatter") == 0) {
      if (pattern.size() == 0) {
        std::cerr << "Pattern needs to have length of at least 1" << std::endl;
        exit(1);
      }
      if (pattern_scatter.size() == 0) {
        std::cerr << "Pattern-Scatter needs to have length of at least 1"
                  << std::endl;
        exit(1);
      }
    } else if (kernel.compare("sg") == 0) {
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

    // Gather and Scatter
    // dense size = pattern.size() * wrap
    // sparse size = max_pattern_val + delta * (count - 1) + 1
    //
    // Concurrent
    // sparse_scatter size = max_pattern_scatter_val + delta_scatter * (count -
    // 1) + 1 sparse_gather size = max_pattern_gather_val + delta_gather *
    // (count - 1) + 1
    //
    // MultiGather
    // dense size = pattern.size() * wrap
    // sparse size = max_pattern_val + delta * (count - 1) + 1
    // assert(pattern.size() > max_pattern_gather_val + 1)
    //
    // MultiScatter
    // dense size = pattern.size() * wrap
    // sparse size = max_pattern_val + delta * (count - 1) + 1
    // assert(pattern.size() > max_pattern_scatter_val + 1)

    if (kernel.compare("sg") == 0) {
      size_t max_pattern_scatter_val = *(std::max_element(
          std::cbegin(pattern_scatter), std::cend(pattern_scatter)));
      size_t max_pattern_gather_val = *(std::max_element(
          std::cbegin(pattern_gather), std::cend(pattern_gather)));
      size_t sparse_scatter_size =
          max_pattern_scatter_val + delta_scatter * (count - 1) + 1;
      size_t sparse_gather_size =
          max_pattern_gather_val + delta_gather * (count - 1) + 1;

      sparse_scatter.resize(sparse_scatter_size);

      for (size_t i = 0; i < sparse_scatter.size(); ++i)
        sparse_scatter[i] = rand();

      sparse_gather.resize(sparse_gather_size);

      for (size_t i = 0; i < sparse_gather.size(); ++i)
        sparse_gather[i] = rand();

      if (verbosity >= 3)
        std::cout << "Pattern Gather Array Size: " << pattern_gather.size()
                  << "Pattern Scatter Array Size: " << pattern_scatter.size()
                  << "\tDelta: " << delta << "\tCount: " << count
                  << "\tWrap: " << wrap
                  << "\tSparse Scatter Array Size: " << sparse_scatter.size()
                  << "\tSparse Gather Array Size: " << sparse_gather.size()
                  << "\tMax Pattern Scatter Val: " << max_pattern_scatter_val
                  << "\tMax Pattern Gather Val: " << max_pattern_gather_val
                  << std::endl;
    } else {
      const size_t max_pattern_val =
          *(std::max_element(std::begin(pattern), std::end(pattern)));
      const size_t dense_size = pattern.size() * wrap;
      const size_t sparse_size = max_pattern_val + delta * (count - 1) + 1;

      dense.resize(dense_size);

      for (size_t i = 0; i < dense.size(); ++i)
        dense[i] = rand();

      sparse.resize(sparse_size);

      for (size_t i = 0; i < sparse.size(); ++i)
        sparse[i] = rand();

      if (kernel.compare("multiscatter") == 0) {
        const size_t max_pattern_scatter_val = *(std::max_element(
            std::begin(pattern_scatter), std::end(pattern_scatter)));
        if (pattern.size() <= max_pattern_scatter_val + 1) {
          std::cerr << "Pattern only has length " << pattern.size()
                    << " but needs to have length of at least "
                       "max_pattern_scatter_val + 1 = "
                    << max_pattern_scatter_val + 1 << std::endl;
          exit(1);
        }
      }

      if (kernel.compare("multigather") == 0) {
        const size_t max_pattern_gather_val = *(std::max_element(
            std::begin(pattern_gather), std::end(pattern_gather)));
        if (pattern.size() <= max_pattern_gather_val + 1) {
          std::cerr << "Pattern only has length " << pattern.size()
                    << " but needs to have length of at least "
                       "max_pattern_gather_val + 1 = "
                    << max_pattern_gather_val + 1 << std::endl;
          exit(1);
        }
      }

      if (verbosity >= 3) {
        std::cout << "Pattern Array Size: " << pattern.size()
                  << "\tDelta: " << delta << "\tCount: " << count
                  << "\tWrap: " << wrap
                  << "\tDense Array Size: " << dense.size()
                  << "\tSparse Array Size: " << sparse.size()
                  << "\tMax Pattern Val: " << max_pattern_val;

        if (kernel.compare("multiscatter") == 0)
          std::cout << "\tMax Pattern Scatter Val: "
                    << *(std::max_element(std::begin(pattern_scatter),
                           std::end(pattern_scatter)));

        if (kernel.compare("multigather") == 0)
          std::cout << "\tMax Pattern Gather Val: "
                    << *(std::max_element(std::begin(pattern_gather),
                           std::end(pattern_gather)));

        std::cout << std::endl;
      }
    }
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
  const size_t delta_gather;
  const std::vector<size_t> deltas_gather;
  const size_t delta_scatter;
  const std::vector<size_t> deltas_scatter;

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

  const bool aggregate;
  const bool compress;
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
  Configuration(const std::string name, const std::string kernel,
      const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const size_t wrap,
      const size_t count, const unsigned long nruns = 10,
      const bool aggregate = false, const bool compress = false,
      const unsigned long verbosity = 3)
      : ConfigurationBase(name, kernel, pattern, pattern_gather,
            pattern_scatter, delta, delta_gather, delta_scatter, wrap, count, 1,
            nruns, aggregate, compress, verbosity) {
    setup();
  };

  void gather(bool timed) {
    size_t pattern_length = pattern.size();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];

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

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];

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

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
            sparse_gather[pattern_gather[j] + delta_gather * i];

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

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] =
            sparse[pattern[pattern_gather[j]] + delta * i];

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

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[pattern_scatter[j]] + delta * i] =
            dense[j + pattern_length * (i % wrap)];

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
  Configuration(const std::string name, const std::string kernel,
      const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const size_t wrap,
      const size_t count, const int nthreads, const unsigned long nruns = 10,
      const bool aggregate = false, const bool compress = false,
      const unsigned long verbosity = 3)
      : ConfigurationBase(name, kernel, pattern, pattern_gather,
            pattern_scatter, delta, delta_gather, delta_scatter, wrap, count,
            nthreads, nruns, aggregate, compress, verbosity) {
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
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];

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
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];

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
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
            sparse_gather[pattern_gather[j] + delta_gather * i];

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
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] =
            sparse[pattern[pattern_gather[j]] + delta * i];

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
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[pattern_scatter[j]] + delta * i] =
            dense[j + pattern_length * (i % wrap)];

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
  Configuration(const std::string name, const std::string kernel,
      const std::vector<size_t> pattern,
      const std::vector<size_t> pattern_gather,
      const std::vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const size_t wrap,
      const size_t count, const unsigned long nruns = 10,
      const bool aggregate = false, const bool compress = false,
      const unsigned long verbosity = 3)
      : ConfigurationBase(name, kernel, pattern, pattern_gather,
            pattern_scatter, delta, delta_gather, delta_scatter, wrap, count, 1,
            nruns, aggregate, compress, verbosity) {
    setup();
  };

  ~Configuration() {
    std::cout << "Deleting Configuration" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_pattern);
    cudaFree(dev_pattern_gather);
    cudaFree(dev_pattern_scatter);

    cudaFree(dev_sparse);
    cudaFree(dev_sparse_gather);
    cudaFree(dev_sparse_scatter);

    cudaFree(dev_dense);
  }

  int run(bool timed) {
    ConfigurationBase::run(timed);

    if (verbosity >= 3)
      std::cout << "Copying Vectors back to CPU" << std::endl;

    cudaMemcpy(sparse.data(), dev_sparse, sizeof(double) * sparse.size(),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(sparse_gather.data(), dev_sparse_gather,
        sizeof(double) * sparse_gather.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sparse_scatter.data(), dev_sparse_scatter,
        sizeof(double) * sparse_scatter.size(), cudaMemcpyDeviceToHost);

    cudaMemcpy(dense.data(), dev_dense, sizeof(double) * dense.size(),
        cudaMemcpyDeviceToHost);

    if (verbosity >= 3)
      std::cout << "Synchronizing CUDA Device" << std::endl;

    cudaDeviceSynchronize();

    if (verbosity >= 3) {
      if (kernel.compare("sg") == 0) {
        std::cout << "Pattern Gather: ";
        for (size_t val : pattern_gather)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Pattern Scatter: ";
        for (size_t val : pattern_scatter)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Sparse Gather: ";
        for (double val : sparse_gather)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Sparse Scatter: ";
        for (double val : sparse_scatter)
          std::cout << val << " ";
        std::cout << std::endl;
      } else {
        std::cout << "Pattern: ";
        for (size_t val : pattern)
          std::cout << val << " ";
        std::cout << std::endl;

        if (kernel.compare("multiscatter") == 0) {
          std::cout << "Pattern Scatter: ";
          for (size_t val : pattern_scatter)
            std::cout << val << " ";
          std::cout << std::endl;
        }

        if (kernel.compare("multigather") == 0) {
          std::cout << "Pattern Gather: ";
          for (size_t val : pattern_gather)
            std::cout << val << " ";
          std::cout << std::endl;
        }

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
    int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_scatter_gather_wrapper(dev_pattern_scatter, dev_sparse_scatter,
        dev_pattern_gather, dev_sparse_gather, pattern_length);

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_gather(bool timed) {
    int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_multi_gather_wrapper(
        dev_pattern, dev_pattern_gather, dev_sparse, dev_dense, pattern_length);

    if (timed) {
      timer.stop();
      time_seconds = timer.seconds();
    }
  }

  void multi_scatter(bool timed) {
    int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (timed)
      timer.start();

    cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter, dev_sparse,
        dev_dense, pattern_length);

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
    cudaMalloc(
        (void **)&dev_pattern_gather, sizeof(size_t) * pattern_gather.size());
    cudaMalloc(
        (void **)&dev_pattern_scatter, sizeof(size_t) * pattern_scatter.size());
    cudaMalloc((void **)&dev_sparse, sizeof(double) * sparse.size());
    cudaMalloc(
        (void **)&dev_sparse_gather, sizeof(double) * sparse_gather.size());
    cudaMalloc(
        (void **)&dev_sparse_scatter, sizeof(double) * sparse_scatter.size());
    cudaMalloc((void **)&dev_dense, sizeof(double) * dense.size());

    if (verbosity >= 3)
      std::cout << "Copying Vectors on to CUDA Device" << std::endl;

    cudaMemcpy(dev_pattern, pattern.data(), sizeof(size_t) * pattern.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pattern_gather, pattern_gather.data(),
        sizeof(size_t) * pattern_gather.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pattern_scatter, pattern_scatter.data(),
        sizeof(size_t) * pattern_scatter.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sparse, sparse.data(), sizeof(double) * sparse.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sparse_gather, sparse_gather.data(),
        sizeof(double) * sparse_gather.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sparse_scatter, sparse_scatter.data(),
        sizeof(double) * sparse_scatter.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dense, dense.data(), sizeof(double) * dense.size(),
        cudaMemcpyHostToDevice);

    if (verbosity >= 3)
      std::cout << "Synchronizing CUDA Device" << std::endl;

    cudaDeviceSynchronize();

    if (verbosity >= 3) {
      if (kernel.compare("sg") == 0) {
        std::cout << "Pattern Gather: ";
        for (size_t val : pattern_gather)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Pattern Scatter: ";
        for (size_t val : pattern_scatter)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Sparse Gather: ";
        for (double val : sparse_gather)
          std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Sparse Scatter: ";
        for (double val : sparse_scatter)
          std::cout << val << " ";
        std::cout << std::endl;
      } else {
        std::cout << "Pattern: ";
        for (size_t val : pattern)
          std::cout << val << " ";
        std::cout << std::endl;

        if (kernel.compare("multiscatter") == 0) {
          std::cout << "Pattern Scatter: ";
          for (size_t val : pattern_scatter)
            std::cout << val << " ";
          std::cout << std::endl;
        }

        if (kernel.compare("multigather") == 0) {
          std::cout << "Pattern Gather: ";
          for (size_t val : pattern_gather)
            std::cout << val << " ";
          std::cout << std::endl;
        }

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
