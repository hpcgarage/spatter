/*!
  \file Configuration.hh
*/

#ifndef SPATTER_CONFIGURATION_HH
#define SPATTER_CONFIGURATION_HH

#ifdef USE_CUDA
#include "CudaBackend.hh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <experimental/iterator>
#include <functional>
#include <ostream>
#include <sstream>

#include "Spatter/SpatterTypes.hh"
#include "Spatter/Timer.hh"

namespace Spatter {

class ConfigurationBase {
public:
  ConfigurationBase(const std::string kernel, const std::vector<size_t> pattern,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : kernel(kernel), pattern(pattern), nruns(nruns), verbosity(verbosity) {}

  ~ConfigurationBase() = default;

  virtual int run(bool timed) = 0;
  virtual void gather(bool timed) = 0;
  virtual void scatter(bool timed) = 0;

  virtual void report() = 0;

private:
  virtual void setup() = 0;

public:
  std::string kernel;
  const std::vector<size_t> pattern;
  std::vector<double> sparse;
  std::vector<double> dense;

  const unsigned long nruns;
  const unsigned long verbosity;

  Spatter::Timer timer;
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
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, nruns, verbosity) {
    setup();
  };

  int run(bool timed) {
    if (kernel.compare("gather") == 0)
      gather(timed);
    else if (kernel.compare("scatter") == 0)
      scatter(timed);
    else {
      std::cerr << "Invalid Kernel Type" << std::endl;
      return -1;
    }

    return 0;
  }

  void gather(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Gather Serial Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      dense[i] = sparse[pattern[i]];

    if (timed)
      timer.stop();
  }

  void scatter(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Scatter Serial Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      sparse[pattern[i]] = dense[i];

    if (timed)
      timer.stop();
  }

  void report() {
    std::cout << "Spatter Serial Report" << std::endl;
    std::cout << nruns * pattern.size() * sizeof(size_t) << " Total Bytes Moved"
              << std::endl;
    std::cout << pattern.size() * sizeof(size_t) << " Bytes Moved per Run"
              << std::endl;
    std::cout << nruns << " Runs took " << std::fixed << timer.seconds()
              << " Seconds" << std::endl;
    std::cout << "Average Bandwidth: "
              << (double)(nruns * pattern.size() * sizeof(size_t)) /
            timer.seconds() / 1000000.
              << " MB/s" << std::endl;
  }

private:
  void setup() {
    if (verbosity >= 3)
      std::cout << "Spatter Serial Setup" << std::endl;

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
};

#ifdef USE_OPENMP
template <> class Configuration<Spatter::OpenMP> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, nruns, verbosity){setup()};

  int run(bool timed) {
    if (kernel.compare("gather") == 0)
      gather(timed);
    else if (kernel.compare("scatter") == 0)
      scatter(timed);
    else {
      std::cerr << "Invalid Kernel Type" << std::endl;
      return -1;
    }

    return 0;
  }

  void gather(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Gather OpenMP Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      dense[i] = sparse[pattern[i]];

    if (timed)
      timer.stop();
  }

  void scatter(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Scatter OpenMP Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      sparse[pattern[i]] = dense[i];

    if (timed)
      timer.stop();
  }

  void report() {
    std::cout << "Spatter OpenMP Report" << std::endl;
    std::cout << nruns * pattern.size() * sizeof(size_t) << " Total Bytes Moved"
              << std::endl;
    std::cout << pattern.size() * sizeof(size_t) << " Bytes Moved per Run"
              << std::endl;
    std::cout << nruns << " Runs took " << std::fixed << timer.seconds()
              << " Seconds" << std::endl;
    std::cout << "Average Bandwidth: "
              << (double)(nruns * pattern.size() * sizeof(size_t)) /
            timer.seconds() / 1000000.
              << " MB/s" << std::endl;
  }

private:
  void setup() {
    if (verbosity >= 3)
      std::cout << "Spatter OpenMP Setup" << std::endl;

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
};
#endif

#ifdef USE_CUDA
template <> class Configuration<Spatter::CUDA> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, nruns, verbosity) {
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
    if (kernel.compare("gather") == 0)
      gather(timed);
    else if (kernel.compare("scatter") == 0)
      scatter(timed);
    else {
      std::cerr << "Invalid Kernel Type" << std::endl;
      return -1;
    }

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
    if (verbosity >= 3)
      std::cout << "Spatter Gather CUDA Running" << std::endl;

    cudaDeviceSynchronize();

    if (timed)
      cudaEventRecord(start);

    int pattern_length = static_cast<int>(pattern.size());
    cuda_gather_wrapper(dev_pattern, dev_sparse, dev_dense, pattern_length);

    /*
        for (size_t i = 0; i < pattern.size(); ++i)
          dense[i] = sparse[pattern[i]];
    */

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
    if (verbosity >= 3)
      std::cout << "Spatter Scatter CUDA Running" << std::endl;

    cudaDeviceSynchronize();

    if (timed)
      cudaEventRecord(start);

    int pattern_length = static_cast<int>(pattern.size());
    cuda_scatter_wrapper(dev_pattern, dev_sparse, dev_dense, pattern_length);

    /*
        for (size_t i = 0; i < pattern.size(); ++i)
          sparse[pattern[i]] = dense[i];
    */
    if (timed) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
    } else
      cudaDeviceSynchronize();

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_seconds += ((double)time_ms / 1000.0);
  }

  void report() {
    std::cout << "Spatter CUDA Report" << std::endl;
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

private:
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

    if (verbosity >= 3)
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDense Array Size: " << dense.size()
                << "\tSparse Array Size: " << sparse.size()
                << "\tMax Pattern Val: " << max_pattern_val << std::endl;

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
  double *dev_sparse;
  double *dev_dense;

  double time_seconds;
  cudaEvent_t start;
  cudaEvent_t stop;
};
#endif

} // namespace Spatter

#endif
