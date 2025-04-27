/*!
  \file Configuration.cc
*/

#include <numeric>
#include <atomic>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "Configuration.hh"
#include "OneAPIBackend.hh"

using namespace sycl;
using namespace dpct;

namespace Spatter {

ConfigurationBase::ConfigurationBase(const size_t id, const std::string name,
    std::string k, const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const int nthreads, const unsigned long nruns, const bool aggregate,
    const bool atomic, const unsigned long verbosity)
    : id(id), name(name), kernel(k), pattern(pattern),
      pattern_gather(pattern_gather), pattern_scatter(pattern_scatter),
      sparse(sparse), dev_sparse(dev_sparse), sparse_size(sparse_size),
      sparse_gather(sparse_gather), dev_sparse_gather(dev_sparse_gather),
      sparse_gather_size(sparse_gather_size), sparse_scatter(sparse_scatter),
      dev_sparse_scatter(dev_sparse_scatter),
      sparse_scatter_size(sparse_scatter_size), dense(dense),
      dense_perthread(dense_perthread), dev_dense(dev_dense),
      dense_size(dense_size), delta(delta), delta_gather(delta_gather),
      delta_scatter(delta_scatter), seed(seed), wrap(wrap), count(count),
      shmem(shared_mem), local_work_size(local_work_size),
      omp_threads(nthreads), nruns(nruns), aggregate(aggregate), atomic(atomic),
      verbosity(verbosity), time_seconds(nruns, 0) {
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
      [](unsigned char c) { return std::tolower(c); });
}

ConfigurationBase::~ConfigurationBase() = default;

int ConfigurationBase::run(bool timed, unsigned long run_id) {
  if (kernel.compare("gather") == 0)
    gather(timed, run_id);
  else if (kernel.compare("scatter") == 0)
    scatter(timed, run_id);
  else if (kernel.compare("sg") == 0)
    scatter_gather(timed, run_id);
  else if (kernel.compare("multigather") == 0)
    multi_gather(timed, run_id);
  else if (kernel.compare("multiscatter") == 0)
    multi_scatter(timed, run_id);
  else {
    std::cerr << "Invalid Kernel Type" << std::endl;
    return -1;
  }

  return 0;
}

void ConfigurationBase::report() {
  size_t bytes_moved = 0;

  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * sizeof(size_t);

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * sizeof(size_t);

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * sizeof(size_t);

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * sizeof(size_t);

#ifdef USE_MPI
  int numpes = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numpes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<unsigned long long> vector_bytes_per_run(numpes, 0);
  MPI_Gather(&bytes_moved, 1, MPI_UNSIGNED_LONG_LONG,
      vector_bytes_per_run.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
      MPI_COMM_WORLD);

  assert(nruns == time_seconds.size());
  std::vector<double> total_time_seconds(nruns, 0.0);
  MPI_Allreduce(time_seconds.data(), total_time_seconds.data(),
      static_cast<int>(nruns), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  long int index = std::distance(total_time_seconds.begin(),
      std::min_element(total_time_seconds.begin(), total_time_seconds.end()));
  assert(index >= 0);
  size_t min_index = static_cast<size_t>(index);

  double mpi_minimum_time = time_seconds[min_index];
  std::vector<double> vector_minimum_time(numpes, 0.0);
  MPI_Gather(&mpi_minimum_time, 1, MPI_DOUBLE, vector_minimum_time.data(), 1,
      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double mpi_maximum_bandwidth =
      static_cast<double>(bytes_per_run) / mpi_minimum_time / 1000000.0;
  std::vector<double> vector_maximum_bandwidth(numpes, 0.0);
  MPI_Gather(&mpi_maximum_bandwidth, 1, MPI_DOUBLE,
      vector_maximum_bandwidth.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
    print_mpi(
        vector_bytes_per_run, vector_minimum_time, vector_maximum_bandwidth);
#else
  double min_time = *std::min_element(time_seconds.begin(), time_seconds.end());
  double bandwidth = static_cast<double>(bytes_moved) / min_time / 1000000.0;

  print_no_mpi(bytes_moved, min_time, bandwidth);
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
    size_t sparse_scatter_size_ =
        max_pattern_scatter_val + delta_scatter * (count - 1) + 1;
    size_t sparse_gather_size_ =
        max_pattern_gather_val + delta_gather * (count - 1) + 1;

    if (sparse_scatter_size < sparse_scatter_size_)
      sparse_scatter_size = sparse_scatter_size_;

    if (sparse_gather_size < sparse_gather_size_)
      sparse_gather_size = sparse_gather_size_;

    if (verbosity >= 3)
      std::cout << "Pattern Gather Array Size: " << pattern_gather.size()
                << "Pattern Scatter Array Size: " << pattern_scatter.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap
                << "\tSparse Scatter Array Size: " << sparse_scatter_size
                << "\tSparse Gather Array Size: " << sparse_gather_size
                << "\tMax Pattern Scatter Val: " << max_pattern_scatter_val
                << "\tMax Pattern Gather Val: " << max_pattern_gather_val
                << std::endl;
  } else {
    const size_t max_pattern_val =
        *(std::max_element(std::begin(pattern), std::end(pattern)));
    const size_t dense_size_ = pattern.size() * wrap;
    const size_t sparse_size_ = max_pattern_val + delta * (count - 1) + 1;

    if (dense_size < dense_size_)
      dense_size = dense_size_;

    if (sparse_size < sparse_size_)
      sparse_size = sparse_size_;

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
                << "\tWrap: " << wrap << "\tDense Array Size: " << dense_size
                << "\tSparse Array Size: " << sparse_size
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

void ConfigurationBase::print_no_mpi(
    size_t bytes_per_run, double minimum_time, double maximum_bandwidth) {
  std::cout << std::setw(15) << std::left << id << std::setw(15) << std::left
            << bytes_per_run << std::setw(15) << std::left << minimum_time
            << std::setw(15) << std::left << maximum_bandwidth << std::endl;
}

#ifdef USE_MPI
void ConfigurationBase::print_mpi(
    std::vector<unsigned long long> &vector_bytes_per_run,
    std::vector<double> &vector_minimum_time,
    std::vector<double> &vector_maximum_bandwidth) {

  unsigned long long total_bytes = std::accumulate(vector_bytes_per_run.begin(),
      vector_bytes_per_run.end(),
      std::remove_reference_t<decltype(vector_bytes_per_run)>::value_type(0));
  double average_bytes_per_rank = static_cast<double>(total_bytes) /
      static_cast<double>(vector_bytes_per_run.size());

  double total_minimum_time = std::accumulate(vector_minimum_time.begin(),
      vector_minimum_time.end(),
      std::remove_reference_t<decltype(vector_minimum_time)>::value_type(0));
  double average_minimum_time_per_rank =
      total_minimum_time / static_cast<double>(vector_minimum_time.size());

  double total_maximum_bandwidth = std::accumulate(
      vector_maximum_bandwidth.begin(), vector_maximum_bandwidth.end(),
      std::remove_reference_t<decltype(vector_maximum_bandwidth)>::value_type(
          0));
  double average_maximum_bandwidth_per_rank = total_maximum_bandwidth /
      static_cast<double>(vector_maximum_bandwidth.size());

  std::cout << std::setw(15) << std::left << id << std::setw(30) << std::left
            << average_bytes_per_rank << std::setw(30) << std::left
            << total_bytes << std::setw(30) << std::left
            << average_minimum_time_per_rank << std::setw(30) << std::left
            << average_maximum_bandwidth_per_rank << std::setw(30) << std::left
            << total_maximum_bandwidth << std::endl;

  if (verbosity >= 3) {
    std::cout << "\nBytes per rank\n";
    for (unsigned long long bytes : vector_bytes_per_run)
      std::cout << bytes << ' ';
    std::cout << '\n';

    std::cout << "Minimum time per rank(s)\n";
    for (double t : vector_minimum_time)
      std::cout << t << ' ';
    std::cout << '\n';

    std::cout << "Maximum bandwidth per rank(MB/s)\n";
    for (double bw : vector_maximum_bandwidth)
      std::cout << bw << ' ';
    std::cout << std::endl;
  }
}
#endif

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  config_output << "{";

  config_output << "'id': " << config.id << ", ";

  if (config.name.compare("") != 0)
    config_output << "'name': '" << config.name << "', ";

  config_output << "'kernel': '" << config.kernel << "', ";

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

  config_output << "'delta': " << config.delta << ", ";
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
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const unsigned long nruns,
    const bool aggregate, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, 0, 1024, 1, nruns, aggregate, false,
          verbosity) {
  ConfigurationBase::setup();
}

void Configuration<Spatter::Serial>::gather(bool timed, unsigned long run_id) {
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
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::scatter(bool timed, unsigned long run_id) {
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
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::scatter_gather(
    bool timed, unsigned long run_id) {
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
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::multi_gather(
    bool timed, unsigned long run_id) {
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
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::multi_scatter(
    bool timed, unsigned long run_id) {
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
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

#ifdef USE_OPENMP
Configuration<Spatter::OpenMP>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed, wrap,
          count, 0, 1024, nthreads, nruns, aggregate, atomic, verbosity) {
  ConfigurationBase::setup();
}

int Configuration<Spatter::OpenMP>::run(bool timed, unsigned long run_id) {
  omp_set_num_threads(omp_threads);
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::OpenMP>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[j]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}

void Configuration<Spatter::OpenMP>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[j]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    double *tl = sparse_scatter.data() + delta_scatter * i;
    double *sl = sparse_gather.data() + delta_gather * i;

#pragma omp simd
    for (size_t j = 0; j < pattern_length; ++j) {
      tl[pattern_scatter[j]] = sl[pattern_gather[j]];
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[pattern_gather[j]]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}


void Configuration<Spatter::OpenMP>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[pattern_scatter[j]]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
#endif

#ifdef USE_CUDA
Configuration<Spatter::CUDA>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed,
          wrap, count, shared_mem, local_work_size, 1, nruns, aggregate, atomic,
          verbosity) {
  
  setup();
}

Configuration<Spatter::CUDA>::~Configuration() {
  checkCudaErrors(cudaFree(dev_pattern));
  checkCudaErrors(cudaFree(dev_pattern_gather));
  checkCudaErrors(cudaFree(dev_pattern_scatter));

  if (dev_sparse) {
    checkCudaErrors(cudaFree(dev_sparse));
    dev_sparse = nullptr;
  }

  if (dev_sparse_gather) {
    checkCudaErrors(cudaFree(dev_sparse_gather));
    dev_sparse_gather = nullptr;
  }

  if (dev_sparse_scatter) {
    checkCudaErrors(cudaFree(dev_sparse_scatter));
    dev_sparse_scatter = nullptr;
  }

  if (dev_dense) {
    checkCudaErrors(cudaFree(dev_dense));
    dev_dense = nullptr;
  }
}

int Configuration<Spatter::CUDA>::run(bool timed, unsigned long run_id) {
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::CUDA>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_gather_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_atomic_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_scatter_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_gather_atomic_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);
  else
    time_ms = cuda_scatter_gather_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_gather(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_multi_gather_wrapper(dev_pattern, dev_pattern_gather,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_scatter(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms =
        cuda_multi_scatter_atomic_wrapper(dev_pattern, dev_pattern_scatter,
            dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter,
        dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::setup() {
  ConfigurationBase::setup();

  checkCudaErrors(
      cudaMalloc((void **)&dev_pattern, sizeof(size_t) * pattern.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_gather, sizeof(size_t) * pattern_gather.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_scatter, sizeof(size_t) * pattern_scatter.size()));

  checkCudaErrors(cudaMemcpy(dev_pattern, pattern.data(),
      sizeof(size_t) * pattern.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_gather, pattern_gather.data(),
      sizeof(size_t) * pattern_gather.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_scatter, pattern_scatter.data(),
      sizeof(size_t) * pattern_scatter.size(), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaDeviceSynchronize());
}
#endif


#ifdef USE_ONEAPI
Configuration<Spatter::OneAPI>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, size_t &sparse_gather_size,
    aligned_vector<double> &sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense, size_t &dense_size,
    aligned_vector<aligned_vector<double>> &dense_perthread, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const size_t shared_mem,
    const size_t local_work_size, const unsigned long nruns,
    const bool aggregate, const bool atomic, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
      pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
      dev_sparse_gather, sparse_gather_size, sparse_scatter,
      dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
      dev_dense, dense_size, delta, delta_gather, delta_scatter, seed,
      wrap, count, shared_mem, local_work_size, 1, nruns, aggregate, atomic,
      verbosity) {
  setup();
}

Configuration<Spatter::OneAPI>::~Configuration() {
  sycl::free(dev_pattern, dpct::get_default_queue());
  sycl::free(dev_pattern_gather, dpct::get_default_queue());
  sycl::free(dev_pattern_scatter, dpct::get_default_queue());

  sycl::free(dev_sparse, dpct::get_default_queue());
  sycl::free(dev_sparse_gather, dpct::get_default_queue());
  sycl::free(dev_sparse_scatter, dpct::get_default_queue());

  // cudaFree(dev_dense);
  sycl::free(dev_dense, dpct::get_default_queue());
}

int Configuration<Spatter::OneAPI>::run(bool timed, unsigned long run_id) {
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::OneAPI>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = oneapi_gather_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::OneAPI>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = oneapi_scatter_atomic_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = oneapi_scatter_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::OneAPI>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = oneapi_scatter_gather_atomic_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);
  else
    time_ms = oneapi_scatter_gather_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::OneAPI>::multi_gather(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = oneapi_multi_gather_wrapper(dev_pattern, dev_pattern_gather,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::OneAPI>::multi_scatter(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms =
        oneapi_multi_scatter_atomic_wrapper(dev_pattern, dev_pattern_scatter,
            dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = oneapi_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter,
        dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::OneAPI>::setup() {
  ConfigurationBase::setup();

  if (sparse.size() < sparse_size) {
    sparse.resize(sparse_size);

    for (size_t i = 0; i < sparse.size(); ++i)
      sparse[i] = rand();
  }

  if (sparse_gather.size() < sparse_gather_size) {
    sparse_gather.resize(sparse_gather_size);

    for (size_t i = 0; i < sparse_gather.size(); ++i)
      sparse_gather[i] = rand();
  }

  if (sparse_scatter.size() < sparse_scatter_size) {
    sparse_scatter.resize(sparse_scatter_size);

    for (size_t i = 0; i < sparse_scatter.size(); ++i)
      sparse_scatter[i] = rand();
  }

  if (dense.size() < dense_size) {
    dense.resize(dense_size);

    for (size_t i = 0; i < dense.size(); ++i)
      dense[i] = rand();
  }

  // cudaMalloc((void **)&dev_pattern, sizeof(size_t) * pattern.size());
  dev_pattern = (size_t *)sycl::malloc_device(sizeof(size_t) * pattern.size(), dpct::get_default_queue());
  // cudaMalloc( (void **)&dev_pattern_gather, sizeof(size_t) * pattern_gather.size());
  dev_pattern_gather = (size_t *)sycl::malloc_device(sizeof(size_t) * pattern_gather.size(), dpct::get_default_queue());
  // cudaMalloc( (void **)&dev_pattern_scatter, sizeof(size_t) * pattern_scatter.size());
  dev_pattern_scatter = (size_t *)sycl::malloc_device(sizeof(size_t) * pattern_scatter.size(), dpct::get_default_queue());

  // cudaMalloc((void **)&dev_sparse, sizeof(double) * sparse.size());
  dev_sparse = (double *)sycl::malloc_device(sizeof(double) * sparse.size(), dpct::get_default_queue());
  // cudaMalloc( (void **)&dev_sparse_gather, sizeof(double) * sparse_gather.size());
  dev_sparse_gather = (double *)sycl::malloc_device(sizeof(double) * sparse_gather.size(), dpct::get_default_queue());
  // cudaMalloc( (void **)&dev_sparse_scatter, sizeof(double) * sparse_scatter.size());
  dev_sparse_scatter = (double *)sycl::malloc_device(sizeof(double) * sparse_scatter.size(), dpct::get_default_queue());
  // cudaMalloc((void **)&dev_dense, sizeof(double) * dense.size());
  dev_dense = (double *)sycl::malloc_device(sizeof(double) * dense.size(), dpct::get_default_queue());

  // cudaMemcpy(dev_pattern, pattern.data(), sizeof(size_t) * pattern.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_pattern, pattern.data(), sizeof(size_t) * pattern.size()).wait();
  // cudaMemcpy(dev_pattern_gather, pattern_gather.data(), sizeof(size_t) * pattern_gather.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_pattern_gather, pattern_gather.data(), sizeof(size_t) * pattern_gather.size()).wait(); 
  // cudaMemcpy(dev_pattern_scatter, pattern_scatter.data(), sizeof(size_t) * pattern_scatter.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_pattern_scatter, pattern_scatter.data(), sizeof(size_t) * pattern_scatter.size()).wait();

  // cudaMemcpy(dev_sparse, sparse.data(), sizeof(double) * sparse.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_sparse, sparse.data(), sizeof(size_t) * sparse.size()).wait();
  // cudaMemcpy(dev_sparse_gather, sparse_gather.data(), sizeof(double) * sparse_gather.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_sparse_gather, sparse_gather.data(), sizeof(size_t) * sparse_gather.size()).wait();
  // cudaMemcpy(dev_sparse_scatter, sparse_scatter.data(), sizeof(double) * sparse_scatter.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_sparse_scatter, sparse_scatter.data(), sizeof(size_t) * sparse_scatter.size()).wait();
  // cudaMemcpy(dev_dense, dense.data(), sizeof(double) * dense.size(), cudaMemcpyHostToDevice);
  dpct::get_default_queue().memcpy(dev_dense, dense.data(), sizeof(size_t) * dense.size()).wait();

  // cudaDeviceSynchronize();
  dpct::get_default_queue().wait();
}
#endif

} // namespace Spatter
