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
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "AlignedAllocator.hh"
#include "SpatterTypes.hh"
#include "Timer.hh"

#define ALIGN 64
template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, ALIGN>>;

namespace Spatter {

class ConfigurationBase {
public:
  ConfigurationBase(const size_t id, const std::string name, std::string k,
      const aligned_vector<size_t> pattern,
      const aligned_vector<size_t> pattern_gather,
      const aligned_vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const int seed,
      const size_t wrap, const size_t count, const int nthreads,
      const unsigned long nruns, const bool aggregate, const bool compress,
      const unsigned long verbosity);

  virtual ~ConfigurationBase();

  virtual int run(bool timed);

  virtual void gather(bool timed) = 0;
  virtual void scatter(bool timed) = 0;
  virtual void scatter_gather(bool timed) = 0;
  virtual void multi_gather(bool timed) = 0;
  virtual void multi_scatter(bool timed) = 0;

  virtual void report();

  virtual void setup();

public:
  const size_t id;
  const std::string name;

  std::string kernel;
  const aligned_vector<size_t> pattern;
  const aligned_vector<size_t> pattern_gather;
  const aligned_vector<size_t> pattern_scatter;

  aligned_vector<double> sparse;
  aligned_vector<double> sparse_gather;
  aligned_vector<double> sparse_scatter;

  aligned_vector<double> dense;

  const size_t delta;
  const aligned_vector<size_t> deltas;
  const size_t delta_gather;
  const aligned_vector<size_t> deltas_gather;
  const size_t delta_scatter;
  const aligned_vector<size_t> deltas_scatter;

  int seed;
  const size_t wrap;
  const size_t count;

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

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config);

template <typename Backend> class Configuration : public ConfigurationBase {};

template <> class Configuration<Spatter::Serial> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> pattern,
      const aligned_vector<size_t> pattern_gather,
      const aligned_vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const int seed,
      const size_t wrap, const size_t count, const unsigned long nruns,
      const bool aggregate, const bool compress, const unsigned long verbosity);

  void gather(bool timed);
  void scatter(bool timed);
  void scatter_gather(bool timed);
  void multi_gather(bool timed);
  void multi_scatter(bool timed);
};

#ifdef USE_OPENMP
template <> class Configuration<Spatter::OpenMP> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> pattern,
      const aligned_vector<size_t> pattern_gather,
      aligned_vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const int seed,
      const size_t wrap, const size_t count, const int nthreads,
      const unsigned long nruns, const bool aggregate, const bool compress,
      const unsigned long verbosity);

  int run(bool timed);

  void gather(bool timed);
  void scatter(bool timed);
  void scatter_gather(bool timed);
  void multi_gather(bool timed);
  void multi_scatter(bool timed);
};
#endif

#ifdef USE_CUDA
template <> class Configuration<Spatter::CUDA> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> pattern,
      const aligned_vector<size_t> pattern_gather,
      const aligned_vector<size_t> pattern_scatter, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const int seed,
      const size_t wrap, const size_t count, const unsigned long nruns,
      const bool aggregate, const bool compress, const unsigned long verbosity);

  ~Configuration();

  int run(bool timed);
  void gather(bool timed);
  void scatter(bool timed);
  void scatter_gather(bool timed);
  void multi_gather(bool timed);
  void multi_scatter(bool timed);
  void setup();

public:
  size_t *dev_pattern;
  size_t *dev_pattern_gather;
  size_t *dev_pattern_scatter;

  double *dev_sparse;
  double *dev_sparse_gather;
  double *dev_sparse_scatter;

  double *dev_dense;
};
#endif

} // namespace Spatter

#endif
