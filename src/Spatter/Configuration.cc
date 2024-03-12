/*!
  \file Configuration.cc
*/

#include <numeric>

#include "Configuration.hh"

namespace Spatter {

ConfigurationBase::ConfigurationBase(const size_t id, const std::string name,
    std::string k, const aligned_vector<size_t> pattern,
    const aligned_vector<size_t> pattern_gather,
    const aligned_vector<size_t> pattern_scatter, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool compress,
    const unsigned long verbosity)
    : id(id), name(name), kernel(k), pattern(pattern),
      pattern_gather(pattern_gather), pattern_scatter(pattern_scatter),
      delta(delta), delta_gather(delta_gather), delta_scatter(delta_scatter),
      seed(seed), wrap(wrap), count(count), omp_threads(nthreads), nruns(nruns),
      aggregate(aggregate), compress(compress), verbosity(verbosity),
      time_seconds(0) {
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
      [](unsigned char c) { return std::tolower(c); });
}

ConfigurationBase::~ConfigurationBase() = default;

int ConfigurationBase::run(bool timed) {
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

void ConfigurationBase::report() {
  size_t total_bytes_moved = 0;

  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    total_bytes_moved = nruns * pattern.size() * count * sizeof(size_t);

  if (kernel.compare("sg") == 0)
    total_bytes_moved = nruns * pattern_gather.size() * count * sizeof(size_t);

  if (kernel.compare("multiscatter") == 0)
    total_bytes_moved = nruns * pattern_scatter.size() * count * sizeof(size_t);

  if (kernel.compare("multigather") == 0)
    total_bytes_moved = nruns * pattern_gather.size() * count * sizeof(size_t);

  int bytes_per_run =
      static_cast<int>(total_bytes_moved) / static_cast<int>(nruns);

  double average_time_per_run = time_seconds / (double)nruns;
  double average_bandwidth =
      (double)(bytes_per_run) / average_time_per_run / 1000000.0;

#ifdef USE_MPI
  int numpes = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numpes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> vector_bytes_per_run(numpes, 0);
  MPI_Gather(&bytes_per_run, 1, MPI_INT, vector_bytes_per_run.data(), 1,
      MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> vector_average_time_per_run(numpes, 0.0);
  MPI_Gather(&average_time_per_run, 1, MPI_DOUBLE,
      vector_average_time_per_run.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> vector_average_bandwidth(numpes, 0.0);
  MPI_Gather(&average_bandwidth, 1, MPI_DOUBLE, vector_average_bandwidth.data(),
      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
    print_mpi(vector_bytes_per_run, vector_average_time_per_run,
        vector_average_bandwidth);
#else
  print_no_mpi(bytes_per_run, average_time_per_run, average_bandwidth);
#endif
}

void ConfigurationBase::setup() {
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
    if (pattern_scatter.size() != pattern_gather.size()) {
      std::cerr
          << "Pattern-Scatter needs to be the same length as Pattern-gather"
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
      if (pattern.size() <= max_pattern_scatter_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_scatter_val = "
                  << max_pattern_scatter_val << std::endl;
        exit(1);
      }
    }

    if (kernel.compare("multigather") == 0) {
      const size_t max_pattern_gather_val = *(std::max_element(
          std::begin(pattern_gather), std::end(pattern_gather)));
      if (pattern.size() <= max_pattern_gather_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_gather_val = "
                  << max_pattern_gather_val << std::endl;
        exit(1);
      }
    }

    if (verbosity >= 3) {
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap << "\tDense Array Size: " << dense.size()
                << "\tSparse Array Size: " << sparse.size()
                << "\tMax Pattern Val: " << max_pattern_val;

      if (kernel.compare("multiscatter") == 0)
        std::cout << "\tMax Pattern Scatter Val: "
                  << *(std::max_element(std::begin(pattern_scatter),
                         std::end(pattern_scatter)));

      if (kernel.compare("multigather") == 0)
        std::cout << "\tMax Pattern Gather Val: "
                  << *(std::max_element(
                         std::begin(pattern_gather), std::end(pattern_gather)));

      std::cout << std::endl;
    }
  }
}

void ConfigurationBase::print_no_mpi(size_t bytes_per_run,
    double average_time_per_run, double average_bandwidth) {
  std::cout << std::setw(15) << std::left << id << std::setw(15) << std::left
            << bytes_per_run << std::setw(15) << std::left
            << average_time_per_run << std::setw(15) << std::left
            << average_bandwidth << std::endl;
}

#ifdef USE_MPI
void ConfigurationBase::print_mpi(std::vector<int> &vector_bytes_per_run,
    std::vector<double> &vector_time_per_run,
    std::vector<double> &vector_average_bandwidth) {

  int total_bytes = std::accumulate(vector_bytes_per_run.begin(),
      vector_bytes_per_run.end(),
      std::remove_reference_t<decltype(vector_bytes_per_run)>::value_type(0));
  double average_bytes_per_rank = static_cast<double>(total_bytes) /
      static_cast<double>(vector_bytes_per_run.size());

  double total_time = std::accumulate(vector_time_per_run.begin(),
      vector_time_per_run.end(),
      std::remove_reference_t<decltype(vector_time_per_run)>::value_type(0));
  double average_time_per_rank =
      total_time / static_cast<double>(vector_time_per_run.size());

  double total_average_bandwidth = std::accumulate(
      vector_average_bandwidth.begin(), vector_average_bandwidth.end(),
      std::remove_reference_t<decltype(vector_average_bandwidth)>::value_type(
          0));
  double average_bandwidth_per_rank = total_average_bandwidth /
      static_cast<double>(vector_average_bandwidth.size());

  std::cout << std::setw(15) << std::left << id << std::setw(30) << std::left
            << average_bytes_per_rank << std::setw(30) << std::left
            << total_bytes << std::setw(30) << std::left
            << average_time_per_rank << std::setw(30) << std::left
            << average_bandwidth_per_rank << std::setw(30) << std::left
            << total_average_bandwidth << std::endl;

  if (verbosity >= 3) {
    std::cout << "Bytes per run per rank\n";
    for (int bytes : vector_bytes_per_run)
      std::cout << bytes << ' ';
    std::cout << "\n\n";

    std::cout << "Average time per run per rank(s)\n";
    for (double t : vector_time_per_run)
      std::cout << t << ' ';
    std::cout << "\n\n";

    std::cout << "Average bandwidth per run per rank(MB/s)\n";
    for (double bw : vector_average_bandwidth)
      std::cout << bw << ' ';
    std::cout << std::endl;
  }
}
#endif

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  config_output << "{";

  config_output << "'id: " << config.id << ", ";

  if (config.name.compare("") != 0)
    config_output << "'name': " << config.name << ", ";

  config_output << "'kernel': " << config.kernel << ", ";

  config_output << "'pattern': [";
  std::copy(std::begin(config.pattern), std::end(config.pattern),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-gather': [";
  std::copy(std::begin(config.pattern_gather), std::end(config.pattern_gather),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-scatter': [";
  std::copy(std::begin(config.pattern_scatter),
      std::end(config.pattern_scatter),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'delta': " << config.delta << ",";
  config_output << "'delta-gather': " << config.delta_gather << ", ";
  config_output << "'delta-scatter': " << config.delta_scatter << ", ";

  config_output << "'count': " << config.count << ", ";

  if (config.seed > 0)
    config_output << "'seed': " << config.seed << ", ";

  if (config.aggregate)
    config_output << "'agg (nruns)': " << config.nruns << ", ";

  config_output << "'wrap': " << config.wrap << ", ";

  config_output << "'threads': " << config.omp_threads;

  config_output << "}";
  return out << config_output.str();
}

Configuration<Spatter::Serial>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> pattern,
    const aligned_vector<size_t> pattern_gather,
    const aligned_vector<size_t> pattern_scatter, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const int seed,
    const size_t wrap, const size_t count, const unsigned long nruns,
    const bool aggregate, const bool compress, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, delta, delta_gather, delta_scatter, seed, wrap,
          count, 1, nruns, aggregate, compress, verbosity) {
  ConfigurationBase::setup();
}

void Configuration<Spatter::Serial>::gather(bool timed) {
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

void Configuration<Spatter::Serial>::scatter(bool timed) {
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

void Configuration<Spatter::Serial>::scatter_gather(bool timed) {
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

void Configuration<Spatter::Serial>::multi_gather(bool timed) {
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

void Configuration<Spatter::Serial>::multi_scatter(bool timed) {
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

#ifdef USE_OPENMP
Configuration<Spatter::OpenMP>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> pattern,
    const aligned_vector<size_t> pattern_gather,
    aligned_vector<size_t> pattern_scatter, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool compress,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, delta, delta_gather, delta_scatter, seed, wrap,
          count, nthreads, nruns, aggregate, compress, verbosity) {
  ConfigurationBase::setup();
};

int Configuration<Spatter::OpenMP>::run(bool timed) {
  omp_set_num_threads(omp_threads);
  return ConfigurationBase::run(timed);
}

void Configuration<Spatter::OpenMP>::gather(bool timed) {
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

void Configuration<Spatter::OpenMP>::scatter(bool timed) {
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

void Configuration<Spatter::OpenMP>::scatter_gather(bool timed) {
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

void Configuration<Spatter::OpenMP>::multi_gather(bool timed) {
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

void Configuration<Spatter::OpenMP>::multi_scatter(bool timed) {
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
#endif

#ifdef USE_CUDA
Configuration<Spatter::CUDA>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> pattern,
    const aligned_vector<size_t> pattern_gather,
    const aligned_vector<size_t> pattern_scatter, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const int seed,
    const size_t wrap, const size_t count, const unsigned long nruns,
    const bool aggregate, const bool compress, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, delta, delta_gather, delta_scatter, seed, wrap,
          count, 1, nruns, aggregate, compress, verbosity) {
  setup();
}

Configuration<Spatter::CUDA>::~Configuration() {
  cudaFree(dev_pattern);
  cudaFree(dev_pattern_gather);
  cudaFree(dev_pattern_scatter);

  cudaFree(dev_sparse);
  cudaFree(dev_sparse_gather);
  cudaFree(dev_sparse_scatter);

  cudaFree(dev_dense);
}

int Configuration<Spatter::CUDA>::run(bool timed) {
  ConfigurationBase::run(timed);

  cudaMemcpy(sparse.data(), dev_sparse, sizeof(double) * sparse.size(),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(sparse_gather.data(), dev_sparse_gather,
      sizeof(double) * sparse_gather.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(sparse_scatter.data(), dev_sparse_scatter,
      sizeof(double) * sparse_scatter.size(), cudaMemcpyDeviceToHost);

  cudaMemcpy(dense.data(), dev_dense, sizeof(double) * dense.size(),
      cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  return 0;
}

void Configuration<Spatter::CUDA>::gather(bool timed) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_gather_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  cudaDeviceSynchronize();

  if (timed)
    time_seconds += ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter(bool timed) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_scatter_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  cudaDeviceSynchronize();

  if (timed)
    time_seconds += ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter_gather(bool timed) {
  assert(pattern_scatter.size() == pattern_gather.size());
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_scatter_gather_wrapper(dev_pattern_scatter,
      dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather, pattern_length,
      delta_scatter, delta_gather, wrap, count);

  cudaDeviceSynchronize();

  if (timed)
    time_seconds += ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_gather(bool timed) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_multi_gather_wrapper(dev_pattern, dev_pattern_gather,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  cudaDeviceSynchronize();

  if (timed)
    time_seconds += ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_scatter(bool timed) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  cudaDeviceSynchronize();

  if (timed)
    time_seconds += ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::setup() {
  ConfigurationBase::setup();

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

  cudaDeviceSynchronize();
}
#endif

} // namespace Spatter
