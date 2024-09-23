/*!
  \file Input.hh
*/

#ifndef SPATTER_INPUT_HH
#define SPATTER_INPUT_HH

#include <algorithm>
#include <cctype>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "Configuration.hh"
#include "JSONParser.hh"
#include "PatternParser.hh"
#include "SpatterTypes.hh"

namespace Spatter {
static char *shortargs =
    (char *)"ab:cd:e:f:g:hj:k:l:m:n:o:p:r:s::t:u:v:w:x:y:z:";
const option longargs[] = {{"aggregate", no_argument, nullptr, 'a'},
    {"atomic-writes", required_argument, nullptr, 0},
    {"backend", required_argument, nullptr, 'b'},
    {"compress", no_argument, nullptr, 'c'},
    {"delta", required_argument, nullptr, 'd'},
    {"boundary", required_argument, nullptr, 'e'},
    {"file", required_argument, nullptr, 'f'},
    {"pattern-gather", required_argument, nullptr, 'g'},
    {"help", no_argument, nullptr, 'h'},
    {"pattern-size", required_argument, nullptr, 'j'},
    {"kernel", required_argument, nullptr, 'k'},
    {"count", required_argument, nullptr, 'l'},
    {"shared-memory", required_argument, nullptr, 'm'},
    {"name", required_argument, nullptr, 'n'},
    {"op", required_argument, nullptr, 'o'},
    {"pattern", required_argument, nullptr, 'p'},
    {"runs", required_argument, nullptr, 'r'},
    {"random", optional_argument, nullptr, 's'},
    {"omp-threads", required_argument, nullptr, 't'},
    {"pattern-scatter", required_argument, nullptr, 'u'},
    {"verbosity", required_argument, nullptr, 'v'},
    {"wrap", required_argument, nullptr, 'w'},
    {"delta-gather", required_argument, nullptr, 'x'},
    {"delta-scatter", required_argument, nullptr, 'y'},
    {"local-work-size", required_argument, nullptr, 'z'}};

struct ClArgs {
  std::vector<std::unique_ptr<Spatter::ConfigurationBase>> configs;

  aligned_vector<double> sparse;
  double *dev_sparse;
  size_t sparse_size;

  aligned_vector<double> sparse_gather;
  double *dev_sparse_gather;
  size_t sparse_gather_size;

  aligned_vector<double> sparse_scatter;
  double *dev_sparse_scatter;
  size_t sparse_scatter_size;

  aligned_vector<double> dense;
  aligned_vector<aligned_vector<double>> dense_perthread;
  double *dev_dense;
  size_t dense_size;

  std::string backend;
  bool aggregate;
  bool atomic;
  bool compress;
  unsigned long verbosity;

  void report_header() {
#ifdef USE_MPI
    int numpes = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numpes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cout << std::setw(15) << std::left << "config" << std::setw(30)
                << std::left << "average bytes per rank" << std::setw(30)
                << std::left << "total bytes" << std::setw(30) << std::left
                << "average time per rank(s)" << std::setw(30) << std::left
                << "average bw per rank(MB/s)" << std::setw(30) << std::left
                << "total bw(MB/s)" << std::endl;
    }
#else
    std::cout << std::setw(15) << std::left << "config" << std::setw(15)
              << std::left << "bytes" << std::setw(15) << std::left << "time(s)"
              << std::setw(15) << std::left << "bw(MB/s)" << std::endl;
#endif
  }
};

std::ostream &operator<<(std::ostream &out, const ClArgs &cl) {
  out << "Run Configurations" << std::endl;
  out << "[ ";

  for (size_t i = 0; i < cl.configs.size(); ++i) {
    if (i != 0)
      out << "  ";
    out << *(cl.configs[i].get());
    if (i != cl.configs.size() - 1)
      out << "," << std::endl;
  }

  out << " ]" << std::endl;
  return out << std::endl;
}

void help(char *progname) {
  std::cout << "Spatter\n";
  std::cout << "Usage: " << progname << "\n";
  std::cout << std::left << std::setw(10) << "-a (--aggregate)" << std::setw(40)
            << "Aggregate (default off)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "   (--atomic-writes)"
            << std::setw(40)
            << "Enable atomic writes for CUDA backend (default 0/off) (TODO: "
               "OpenMP atomics)"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-b (--backend)" << std::setw(40)
            << "Backend (default serial)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-c (--compress)" << std::setw(40)
            << " Enable compression of pattern indices" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-d (--delta)" << std::setw(40)
            << "Delta (default 8)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-e (--boundary)" << std::setw(40)
            << " Set Boundary (limits max value of pattern using modulo)"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-f (--file)" << std::setw(40)
            << "Input File" << std::left << "\n";
  std::cout
      << std::left << std::setw(10) << "-g (--pattern-gather)" << std::setw(40)
      << "Set Inner Gather Pattern (Valid with kernel-name: sg, multigather)"
      << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-h (--help)" << std::setw(40)
            << "Print Help Message" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-j (--pattern-size)"
            << std::setw(40)
            << " Set Pattern Size"
               " (truncates pattern to pattern-size)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-k (--kernel)" << std::setw(40)
            << "Kernel (default gather)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-l (--count)" << std::setw(40)
            << "Set Number of Gathers or Scatters to Perform (default 1024)"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-m (--shared-memory)"
            << std::setw(40)
            << "Set Amount of Dummy Shared Memory to Allocate on GPUs"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-n (--name)" << std::setw(40)
            << "Specify the Configuration Name" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-o (--op)" << std::setw(40)
            << "TODO" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-p (--pattern)" << std::setw(40)
            << "Set Pattern" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-r (--runs)" << std::setw(40)
            << "Set Number of Runs (default 10)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-s (--random)" << std::setw(40)
            << "Set Random Seed (default random)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-t (--omp-threads)"
            << std::setw(40)
            << "Set Number of Threads (default 1 if !USE_OPENMP or backend != "
               "openmp or OMP_MAX_THREADS if USE_OPENMP)"
            << std::left << "\n";
  std::cout
      << std::left << std::setw(10) << "-u (--pattern-scatter)" << std::setw(4)
      << "Set Inner Scatter Pattern (Valid with kernel-name: sg, multiscatter)"
      << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-v (--verbosity)" << std::setw(40)
            << "Set Verbosity Level (default 1)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-w (--wrap)" << std::setw(40)
            << "Set Wrap (default 1)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-x (--delta-gather)"
            << std::setw(40) << "Delta (default 8)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-y (--delta-scatter)"
            << std::setw(40) << "Delta (default 8)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-z (--local-work-size)"
            << std::setw(40) << "Set Local Work Size (default 1024)"
            << std::left << "\n";
}

void usage(char *progname) {
  std::cout << "Usage: " << progname
            << "[-a aggregate] [--atomic-writes] [-b backend] [-c compress] "
               "[-d delta] [-e "
               "boundary] [-f input file] [-g inner gather pattern] "
               "[-h "
               "help] [-j pattern-size] [-k kernel] [-l count] [-m "
               "shared-memory] [-n name] [-o op]"
               "[-p pattern] [-r runs] [-s random] [-t nthreads] [-u inner "
               "scatter pattern] [-v "
               "verbosity] [-w wrap] [-z local-work-size]"
            << std::endl;
}

int read_int_arg(std::string cl, int &arg, const std::string &err_msg) {
  int passed_arg;

  try {
    passed_arg = std::stoi(cl);
  } catch (const std::invalid_argument &ia) {
    std::cerr << err_msg << std::endl;
    return -1;
  }

  if (passed_arg < 0) {
    std::cerr << err_msg << std::endl;
    return -1;
  } else {
    arg = passed_arg;
  }

  return 0;
}

int read_ul_arg(std::string cl, size_t &arg, const std::string &err_msg) {
  int64_t passed_arg;

  try {
    passed_arg = std::stoll(cl);
  } catch (const std::invalid_argument &ia) {
    std::cerr << err_msg << std::endl;
    return -1;
  }

  if (passed_arg < 0) {
    std::cerr << err_msg << std::endl;
    return -1;
  } else {
    arg = static_cast<size_t>(passed_arg);
  }

  return 0;
}

int parse_input(const int argc, char **argv, ClArgs &cl) {
  srand(static_cast<unsigned int>(time(nullptr)));

  cl.sparse_size = 0;
  cl.sparse_gather_size = 0;
  cl.sparse_scatter_size = 0;
  cl.dense_size = 0;

  cl.dev_sparse = nullptr;
  cl.dev_sparse_gather = nullptr;
  cl.dev_sparse_scatter = nullptr;
  cl.dev_dense = nullptr;

  cl.backend = "";
  cl.aggregate = false;
  cl.atomic = false;
  cl.compress = false;
  cl.verbosity = 1;

  // In flag alphabetical order
  bool aggregate = cl.aggregate;
  bool atomic = cl.atomic;
  std::string backend = cl.backend;
  bool compress = cl.compress;
  size_t delta = 8;
  size_t boundary = 0;

  bool json = 0;
  std::string json_fname = "";

  aligned_vector<size_t> pattern_gather;
  std::stringstream pattern_gather_string;

  size_t pattern_size = 0;
  std::string kernel = "gather";
  size_t count = 1024;
  size_t shared_mem;
  std::string config_name = "";
  size_t op;

  std::stringstream pattern_string;
  aligned_vector<size_t> pattern;

  unsigned long nruns = 10;
  long int seed = -1;

#ifdef USE_OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif

  std::stringstream pattern_scatter_string;
  aligned_vector<size_t> pattern_scatter;

  unsigned long verbosity = cl.verbosity;
  size_t wrap = 1;
  size_t delta_gather = 8;
  size_t delta_scatter = 8;
  size_t local_work_size = 1024;

  int option_index = 0;
  optind = 1;
  int c;
  while (
      (c = getopt_long(argc, argv, shortargs, longargs, &option_index)) != -1) {
    switch (c) {
    case 0:
      if (longargs[option_index].flag != 0)
        break;
      if (strcmp(longargs[option_index].name, "atomic-writes") == 0) {
        int atomic_val = 0;
        if (read_int_arg(optarg, atomic_val,
                "Parsing Error: Invalid Atomic Write") == -1)
          return -1;
        atomic = (atomic_val > 0) ? true : false;
      }
      break;

    case 'a':
      aggregate = true;
      break;

    case 'b':
      backend = optarg;
      std::transform(backend.begin(), backend.end(), backend.begin(),
          [](unsigned char c) { return std::tolower(c); });

      if ((backend.compare("serial") != 0) &&
          (backend.compare("openmp") != 0) && (backend.compare("cuda") != 0)) {
        std::cerr << "Valid Backends are: serial, openmp, cuda" << std::endl;
        return -1;
      }
      if (backend.compare("openmp") == 0) {
#ifndef USE_OPENMP
        std::cerr << "FAIL - OpenMP Backend is not Enabled" << std::endl;
        return -1;
#endif
      }
      if (backend.compare("cuda") == 0) {
#ifndef USE_CUDA
        std::cerr << "FAIL - CUDA Backend is not Enabled" << std::endl;
        return -1;
#endif
      }
      break;

    case 'c':
      compress = true;
      break;

    case 'd':
      if (read_ul_arg(optarg, delta, "Parsing Error: Invalid Delta") == -1)
        return -1;
      break;

    case 'e':
      if (read_ul_arg(optarg, boundary, "Parsing Error: Invalid Boundary") ==
          -1)
        return -1;
      break;

    case 'f':
      json = 1;
      json_fname = optarg;
      break;

    case 'g':
      pattern_gather_string << optarg;
      if (pattern_parser(pattern_gather_string, pattern_gather, delta_gather) != 0)
        return -1;
      break;

    case 'h':
      help(argv[0]);
      return -1;

    case 'j':
      if (read_ul_arg(optarg, pattern_size,
              "Parsing Error: Invalid Pattern Size") == -1)
        return -1;
      break;

    case 'k':
      kernel = optarg;
      std::transform(kernel.begin(), kernel.end(), kernel.begin(),
          [](unsigned char c) { return std::tolower(c); });

      if ((kernel.compare("gather") != 0) && (kernel.compare("scatter") != 0) &&
          (kernel.compare("sg") != 0) && (kernel.compare("multigather") != 0) &&
          (kernel.compare("multiscatter") != 0)) {
        std::cerr << "Valid Kernels are: gather, scatter, sg, multigather, "
                     "multiscatter"
                  << std::endl;
        return -1;
      }
      break;

    case 'l':
      if (read_ul_arg(optarg, count, "Parsing Error: Invalid Count") == -1)
        return -1;
      break;

    case 'm':
      if (read_ul_arg(
              optarg, shared_mem, "Parsing Error: Invalid Shared Memory") == -1)
        return -1;
      break;

    case 'n':
      config_name = optarg;
      break;

    case 'o':
      if (read_ul_arg(optarg, op, "Parsing Error: Invalid Operation") == -1)
        return -1;
      break;

    case 'p':
      pattern_string << optarg;
      if (pattern_parser(pattern_string, pattern, delta) != 0)
        return -1;
      break;

    case 'r':
      if (read_ul_arg(optarg, nruns, "Parsing Error: Invalid Number of Runs") ==
          -1)
        return -1;
      break;

    case 's':
      if (optarg != nullptr) {
        int seed_arg = -1;
        if (read_int_arg(optarg, seed_arg,
            "Parsing Error: Invalid Random Seed") == -1)
          return -1;

        seed = seed_arg;
      } else {
        seed = time(NULL);
      }
      break;

    case 't':
      if (read_int_arg(optarg, nthreads,
              "Parsing Error: Invalid Number of Threads") == -1)
        return -1;
      break;

    case 'u':
      pattern_scatter_string << optarg;
      if (pattern_parser(pattern_scatter_string, pattern_scatter, delta_scatter) != 0)
        return -1;
      break;

    case 'v':
      if (read_ul_arg(optarg, verbosity,
              "Parsing Error: Invalid Verbosity Level") == -1)
        return -1;
      break;

    case 'w':
      if (read_ul_arg(optarg, wrap, "Parsing Error: Invalid Wrap") == -1)
        return -1;
      break;

    case 'x':
      if (read_ul_arg(optarg, delta_gather,
              "Parsing Error: Invalid Delta Gather") == -1)
        return -1;
      break;

    case 'y':
      if (read_ul_arg(optarg, delta_scatter,
              "Parsing Error: Invalid Delta Scatter") == -1)
        return -1;
      break;

    case 'z':
      if (read_ul_arg(
              optarg, local_work_size, "Parsing Error: Local Work Size") == -1)
        return -1;
      break;

    case '?':
      usage(argv[0]);
      return -1;
    }
  }

  // Set default backend if one was not specified
  if (backend.compare("") == 0) {
    backend = "serial";
    // Assume only one of USE_CUDA and USE_OPENMP can be true at once
#ifdef USE_OPENMP
    backend = "openmp";
#endif
#ifdef USE_CUDA
      backend = "cuda";
#endif
  }

  cl.backend = backend;
  cl.aggregate = aggregate;
  cl.compress = compress;
  cl.verbosity = verbosity;

#ifdef USE_OPENMP
  int max_threads = omp_get_max_threads();
  if (backend.compare("openmp") != 0) {
    std::cerr << "Parsing Warning: Requested threads without specifying the "
                 "backend to be OpenMP. Backend requested was "
              << backend << ". Using 1 OpenMP thread instead" << std::endl;
    nthreads = 1;
  } else {
    if (nthreads > max_threads) {
      std::cerr << "Parsing Warning: Too many OpenMP threads requested. Using "
                   "OMP_MAX_THREADS instead"
                << std::endl;
      nthreads = max_threads;
    }

    if (nthreads == 0) {
      std::cerr << "Parsing Warning: 0 OpenMP threads requested. Using "
                   "OMP_MAX_THREADS instead"
                << std::endl;
      nthreads = max_threads;
    }

    if (nthreads < 0) {
      std::cerr << "Parsing Warning: Negative OpenMP threads requested. Using "
                   "OMP_MAX_THREADS instead"
                << std::endl;
      nthreads = max_threads;
    }
  }
#else
  if (nthreads != 1) {
    std::cerr << "Compiled without OpenMP support, but requested something "
                 "other than 1 OpenMP thread. Using 1 thread instead"
              << std::endl;
    nthreads = 1;
  }
#endif

  if (pattern_size > 0) {
    if (pattern.size() > 0) {
      if (truncate_pattern(pattern, pattern_size) == -1) {
        std::cerr << "Truncating pattern to size " << pattern_size
                  << " failed" << std::endl;
        return -1;
      }
    }

    if (pattern_gather.size() > 0) {
      if (truncate_pattern(pattern_gather, pattern_size) == -1) {
        std::cerr << "Truncating pattern_gather to size " << pattern_size
                  << " failed" << std::endl;
        return -1;
      }
    }

    if (pattern_scatter.size() > 0) {
      if (truncate_pattern(pattern_scatter, pattern_size) == -1) {
        std::cerr << "Truncating pattern_scatter to size " << pattern_size
                  << " failed" << std::endl;
        return -1;
      }
    }
  }


  if (pattern.size() > 0) {
    if (remap_pattern(pattern, boundary, 1) > boundary) {
      std::cerr << "Re-mapping pattern to have maximum value of " << boundary
                << "failed" << std::endl;
      return -1;
    }
  }

  if (pattern_gather.size() > 0) {
    if (remap_pattern(pattern_gather, boundary, 1) > boundary) {
      std::cerr << "Re-mapping pattern_gather to have maximum value of "
                << boundary << "failed" << std::endl;
      return -1;
    }
  }

  if (pattern_scatter.size() > 0) {
    if (remap_pattern(pattern_scatter, boundary, 1) > boundary) {
      std::cerr << "Re-mapping pattern_scatter to have maximum value of "
                << boundary << "failed" << std::endl;
      return -1;
    }
  }

  if (compress) {
    if (pattern.size() > 0)
      compress_pattern(pattern);

    if (pattern_gather.size() > 0)
      compress_pattern(pattern_gather);

    if (pattern_scatter.size() > 0)
      compress_pattern(pattern_scatter);
  }

  if (!json) {
    std::unique_ptr<Spatter::ConfigurationBase> c;
    if (backend.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(0,
          config_name, kernel, pattern, pattern_gather, pattern_scatter,
          cl.sparse, cl.dev_sparse, cl.sparse_size, cl.sparse_gather,
          cl.dev_sparse_gather, cl.sparse_gather_size, cl.sparse_scatter,
          cl.dev_sparse_scatter, cl.sparse_scatter_size, cl.dense,
          cl.dense_perthread, cl.dev_dense, cl.dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, nruns, aggregate, verbosity);
#ifdef USE_OPENMP
    else if (backend.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(0,
          config_name, kernel, pattern, pattern_gather, pattern_scatter,
          cl.sparse, cl.dev_sparse, cl.sparse_size, cl.sparse_gather,
          cl.dev_sparse_gather, cl.sparse_gather_size, cl.sparse_scatter,
          cl.dev_sparse_scatter, cl.sparse_scatter_size, cl.dense,
          cl.dense_perthread, cl.dev_dense, cl.dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, nthreads, nruns, aggregate, atomic,
          verbosity);
#endif
#ifdef USE_CUDA
    else if (backend.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(0,
          config_name, kernel, pattern, pattern_gather, pattern_scatter,
          cl.sparse, cl.dev_sparse, cl.sparse_size, cl.sparse_gather,
          cl.dev_sparse_gather, cl.sparse_gather_size, cl.sparse_scatter,
          cl.dev_sparse_scatter, cl.sparse_scatter_size, cl.dense,
          cl.dense_perthread, cl.dev_dense, cl.dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, shared_mem, local_work_size, nruns,
          aggregate, atomic, verbosity);
#endif
    else {
      std::cerr << "Invalid Backend " << backend << std::endl;
      return -1;
    }

    cl.configs.push_back(std::move(c));
  } else {
    Spatter::JSONParser json_file = Spatter::JSONParser(json_fname, cl.sparse,
        cl.dev_sparse, cl.sparse_size, cl.sparse_gather, cl.dev_sparse_gather,
        cl.sparse_gather_size, cl.sparse_scatter, cl.dev_sparse_scatter,
        cl.sparse_scatter_size, cl.dense, cl.dense_perthread, cl.dev_dense,
        cl.dense_size, backend, aggregate, atomic, compress, shared_mem,
        nthreads, verbosity);

    for (size_t i = 0; i < json_file.size(); ++i) {
      std::unique_ptr<Spatter::ConfigurationBase> c = json_file[i];
      cl.configs.push_back(std::move(c));
    }
  }

  if (cl.sparse.size() < cl.sparse_size) {
    cl.sparse.resize(cl.sparse_size);

    for (size_t i = 0; i < cl.sparse.size(); ++i)
      cl.sparse[i] = rand();
  }

  if (cl.sparse_gather.size() < cl.sparse_gather_size) {
    cl.sparse_gather.resize(cl.sparse_gather_size);

    for (size_t i = 0; i < cl.sparse_gather.size(); ++i)
      cl.sparse_gather[i] = rand();
  }

  if (cl.sparse_scatter.size() < cl.sparse_scatter_size) {
    cl.sparse_scatter.resize(cl.sparse_scatter_size);

    for (size_t i = 0; i < cl.sparse_scatter.size(); ++i)
      cl.sparse_scatter[i] = rand();
  }

  if (cl.dense.size() < cl.dense_size) {
    cl.dense.resize(cl.dense_size);

    for (size_t i = 0; i < cl.dense.size(); ++i)
      cl.dense[i] = rand();
  }

#ifdef USE_OPENMP
  if (backend.compare("openmp") == 0) {
    cl.dense_perthread.resize(nthreads);

    for (int j = 0; j < nthreads; ++j) {
      cl.dense_perthread[j].resize(cl.dense_size);

      for (size_t i = 0; i < cl.dense_perthread[j].size(); ++i)
        cl.dense_perthread[j][i] = rand();
    }
  }
#endif
#ifdef USE_CUDA
  if (backend.compare("cuda") == 0) {
    checkCudaErrors(cudaMalloc((void **)&cl.dev_sparse,
        sizeof(double) * cl.sparse.size()));
    checkCudaErrors(cudaMalloc((void **)&cl.dev_sparse_gather,
        sizeof(double) * cl.sparse_gather.size()));
    checkCudaErrors(cudaMalloc((void **)&cl.dev_sparse_scatter,
        sizeof(double) * cl.sparse_scatter.size()));
    checkCudaErrors(cudaMalloc((void **)&cl.dev_dense,
        sizeof(double) * cl.dense.size()));

    checkCudaErrors(cudaMemcpy(cl.dev_sparse, cl.sparse.data(),
        sizeof(double) * cl.sparse.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cl.dev_sparse_gather, cl.sparse_gather.data(),
        sizeof(double) * cl.sparse_gather.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cl.dev_sparse_scatter, cl.sparse_scatter.data(),
        sizeof(double) * cl.sparse_scatter.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cl.dev_dense, cl.dense.data(),
        sizeof(double) * cl.dense.size(), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
  }
#endif

  for (auto const &config : cl.configs) {
    if (config->aggregate != aggregate) {
      std::cerr << "Aggregate flag of Config does not match the aggregate flag "
                   "passed to the command line"
                << std::endl;
      return -1;
    }

    // if (config->compress != compress) {
    //   std::cerr << "Compress flag of Config does not match the compress flag "
    //                "passed to the command line"
    //             << std::endl;
    //   return -1;
    // }

    if (config->verbosity != verbosity) {
      std::cerr << "Verbosity level of Config does not match the verbosity "
                   "level passed to the command line"
                << std::endl;
      return -1;
    }
  }

  return 0;
}

} // namespace Spatter

#endif
