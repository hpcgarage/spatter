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

#include "Spatter/Configuration.hh"
#include "Spatter/JSONParser.hh"
#include "Spatter/PatternParser.hh"
#include "Spatter/SpatterTypes.hh"

namespace Spatter {
static char *shortargs =
    (char *)"ab:cd:e:f:g:hj:k:l:m:n:o:p:r:s:t:u:v:w:x:y:z:";
const option longargs[] = {{"aggregate", no_argument, nullptr, 'a'},
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
    {"random", required_argument, nullptr, 's'},
    {"omp-threads", required_argument, nullptr, 't'},
    {"pattern-scatter", required_argument, nullptr, 'u'},
    {"verbosity", required_argument, nullptr, 'v'},
    {"wrap", required_argument, nullptr, 'w'},
    {"delta-gather", required_argument, nullptr, 'x'},
    {"delta-scatter", required_argument, nullptr, 'y'},
    {"local-work-size", required_argument, nullptr, 'z'}};

struct ClArgs {
  std::vector<std::unique_ptr<Spatter::ConfigurationBase>> configs;
};

void help(char *progname) {
  std::cout << "Spatter\n";
  std::cout << "Usage: " << progname << "\n";
  std::cout << std::left << std::setw(10) << "-a (--aggregate)" << std::setw(40)
            << "Aggregate (default off)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-b (--backend)" << std::setw(40)
            << "Backend (default serial)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-c (--compress)" << std::setw(40)
            << "TODO" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-d (--delta)" << std::setw(40)
            << "Delta (default 8)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-e (--boundary)" << std::setw(40)
            << "Set Boundary (i.e. Set max value of pattern array)" << std::left
            << "\n";
  std::cout << std::left << std::setw(10) << "-f (--file)" << std::setw(40)
            << "Input File" << std::left << "\n";
  std::cout
      << std::left << std::setw(10) << "-g (--pattern-gather)" << std::setw(40)
      << "Set Inner Gather Pattern (Valid with kernel-name: sg, multigather)"
      << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-h (--help)" << std::setw(40)
            << "Print Help Message" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-j (--pattern-size)"
            << std::setw(40) << "Set Pattern Size" << std::left << "\n";
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
            << "Set Verbosity Level" << std::left << "\n";
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
            << "[-a aggregate] [-b backend] [-c compress] [-d delta] [-e "
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
  try {
    arg = std::stoi(cl);
  } catch (const std::invalid_argument &ia) {
    std::cerr << err_msg << std::endl;
    return -1;
  }
  return 0;
}

int read_ul_arg(std::string cl, size_t &arg, const std::string &err_msg) {
  try {
    arg = std::stoul(cl);
  } catch (const std::invalid_argument &ia) {
    std::cerr << err_msg << std::endl;
    return -1;
  }
  return 0;
}

size_t remap_pattern(std::vector<size_t> &pattern, const size_t boundary) {
  const size_t pattern_len = pattern.size();
  for (size_t j = 0; j < pattern_len; ++j) {
    pattern[j] = pattern[j] % boundary;
  }

  size_t max_pattern_val = *(std::max_element(pattern.begin(), pattern.end()));
  return max_pattern_val;
}

int parse_input(const int argc, char **argv, ClArgs &cl) {
  // In flag alphabetical order
  bool aggregate = false;
  std::string backend = "serial";
  bool compress = false;
  size_t delta = 8;
  size_t boundary = 0;

  bool json = 0;
  std::string json_fname = "";

  std::vector<size_t> pattern_gather;
  std::stringstream pattern_gather_string;

  size_t pattern_size = 0;
  std::string kernel = "gather";
  size_t count = 1024;
  size_t shared_mem;
  std::string config_name = "";
  size_t op;

  std::stringstream pattern_string;
  std::vector<size_t> pattern;

  unsigned long nruns = 10;
  int seed = -1;

#ifdef USE_OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif

  std::stringstream pattern_scatter_string;
  std::vector<size_t> pattern_scatter;

  unsigned long verbosity = 3;
  size_t wrap = 1;
  size_t delta_gather = 8;
  size_t delta_scatter = 8;
  size_t local_work_size;

  int c;
  while ((c = getopt_long(argc, argv, shortargs, longargs, nullptr)) != -1) {
    switch (c) {
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
        std::cout << "Checking if OpenMP Backend is Enabled" << std::endl;
#ifdef USE_OPENMP
        std::cout << "SUCCESS - OpenMP Backend Enabled" << std::endl;
#else
        std::cerr << "FAIL - OpenMP Backend is not Enabled" << std::endl;
        return -1;
#endif
      }
      if (backend.compare("cuda") == 0) {
        std::cout << "Checking if CUDA Backend is Enabled" << std::endl;
#ifdef USE_CUDA
        std::cout << "SUCCESS - CUDA Backend Enabled" << std::endl;
#else
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
      if (pattern_parser(pattern_gather_string, pattern_gather) != 0)
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
      std::transform(backend.begin(), backend.end(), backend.begin(),
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
      if (pattern_parser(pattern_string, pattern) != 0)
        return -1;
      break;

    case 'r':
      if (read_ul_arg(optarg, nruns, "Parsing Error: Invalid Number of Runs") ==
          -1)
        return -1;
      break;

    case 's':
      if (read_int_arg(optarg, seed, "Parsing Error: Invalid Random Seed") ==
          -1)
        return -1;
      break;

    case 't':
      if (read_int_arg(optarg, nthreads,
              "Parsing Error: Invalid Number of Threads") == -1)
        return -1;
      break;

    case 'u':
      pattern_scatter_string << optarg;
      if (pattern_parser(pattern_scatter_string, pattern_scatter) != 0)
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
    if (pattern.size() > 0)
      pattern.resize(pattern_size);

    if (pattern_gather.size() > 0)
      pattern_gather.resize(pattern_size);

    if (pattern_scatter.size() > 0)
      pattern_scatter.resize(pattern_size);
  }

  if (boundary > 0) {
    if (pattern.size() > 0) {
      if (remap_pattern(pattern, boundary) > boundary) {
        std::cerr << "Re-mapping pattern to have maximum value of " << boundary
                  << "failed" << std::endl;
        return -1;
      }
    }

    if (pattern_gather.size() > 0) {
      if (remap_pattern(pattern_gather, boundary) > boundary) {
        std::cerr << "Re-mapping pattern_gather to have maximum value of "
                  << boundary << "failed" << std::endl;
        return -1;
      }
    }

    if (pattern_scatter.size() > 0) {
      if (remap_pattern(pattern_scatter, boundary) > boundary) {
        std::cerr << "Re-mapping pattern_scatter to have maximum value of "
                  << boundary << "failed" << std::endl;
        return -1;
      }
    }
  }

  if (!json) {
    std::unique_ptr<Spatter::ConfigurationBase> c;
    if (backend.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(config_name,
          kernel, pattern, pattern_gather, pattern_scatter, delta, delta_gather,
          delta_scatter, wrap, count, nruns, aggregate, compress, verbosity);
#ifdef USE_OPENMP
    else if (backend.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(config_name,
          kernel, pattern, pattern_gather, pattern_scatter, delta, delta_gather,
          delta_scatter, wrap, count, nthreads, nruns, aggregate, compress,
          verbosity);
#endif
#ifdef USE_CUDA
    else if (backend.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(config_name,
          kernel, pattern, pattern_gather, pattern_scatter, delta, delta_gather,
          delta_scatter, wrap, count, nruns, aggregate, compress, verbosity);
#endif
    else {
      std::cerr << "Invalid Backend " << backend << std::endl;
      return -1;
    }

    cl.configs.push_back(std::move(c));
  } else {
    Spatter::JSONParser json_file =
        Spatter::JSONParser(json_fname, backend, verbosity);

    for (size_t i = 0; i < json_file.size(); ++i) {
      std::unique_ptr<Spatter::ConfigurationBase> c = json_file[i];
      cl.configs.push_back(std::move(c));
    }
  }

  return 0;
}

} // namespace Spatter

#endif
