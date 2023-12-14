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
static char *shortargs = (char *)"b:d:f:g:hk:l:p:r:s:t:v:w:";
const option longargs[] = {{"backend", required_argument, nullptr, 'b'},
    {"delta", required_argument, nullptr, 'd'},
    {"file", required_argument, nullptr, 'f'},
    {"pattern-gather", required_argument, nullptr, 'g'},
    {"help", no_argument, nullptr, 'h'},
    {"kernel", required_argument, nullptr, 'k'},
    {"count", required_argument, nullptr, 'l'},
    {"pattern", required_argument, nullptr, 'p'},
    {"runs", required_argument, nullptr, 'r'},
    {"pattern-scatter", required_argument, nullptr, 's'},
    {"omp-threads", required_argument, nullptr, 't'},
    {"verbosity", required_argument, nullptr, 'v'},
    {"wrap", required_argument, nullptr, 'w'}};

struct ClArgs {
  std::vector<std::unique_ptr<Spatter::ConfigurationBase>> configs;
};

void help(char *progname) {
  std::cout << "Spatter\n";
  std::cout << "Usage: " << progname << "\n";
  std::cout << std::left << std::setw(10) << "-b (--backend)" << std::setw(40)
            << "Backend (default serial)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-d (--delta)" << std::setw(40)
            << "Delta (default 8)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-f (--file)" << std::setw(40)
            << "Input File" << std::left << "\n";
  std::cout
      << std::left << std::setw(10) << "-g (--pattern-gather)" << std::setw(40)
      << "Set Inner Gather Pattern (Valid with kernel-name: sg, multigather)"
      << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-h (--help)" << std::setw(40)
            << "Print Help Message" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-k (--kernel)" << std::setw(40)
            << "Kernel (default gather)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-l (--count)" << std::setw(40)
            << "Set Number of Gathers or Scatters to Perform (default 1024)"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-p (--pattern)" << std::setw(40)
            << "Set Pattern" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-r (--runs)" << std::setw(40)
            << "Set Number of Runs (default 10)" << std::left << "\n";
  std::cout
      << std::left << std::setw(10) << "-s (--pattern-scatter)" << std::setw(4)
      << "Set Inner Scatter Pattern (Valid with kernel-name: sg, multiscatter)"
      << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-t (--omp-threads)"
            << std::setw(40)
            << "Set Number of Threads (default 1 if !USE_OPENMP or backend != "
               "openmp or OMP_MAX_THREADS if USE_OPENMP)"
            << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-v (--verbosity)" << std::setw(40)
            << "Set Verbosity Level" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-w (--wrap)" << std::setw(40)
            << "Set Wrap (default 1)" << std::left << "\n";
}

void usage(char *progname) {
  std::cout
      << "Usage: " << progname
      << "[-b backend] [-d delta] [-f input file] [-g inner gather pattern] "
         "[-h "
         "help] [-k kernel] [-l count] "
         "[-p pattern] [-r runs] [-s inner scatter pattern] [-t nthreads] [-v "
         "verbosity] [-w wrap]"
      << std::endl;
}

int parse_input(const int argc, char **argv, ClArgs &cl) {
  int c;
  std::stringstream pattern_string;
  std::stringstream pattern_gather_string;
  std::stringstream pattern_scatter_string;

  std::vector<size_t> pattern;
  std::vector<size_t> pattern_gather;
  std::vector<size_t> pattern_scatter;

  size_t count = 1024;
  size_t delta = 8;
  size_t wrap = 1;

  std::string backend = "serial";
  std::string kernel = "gather";

  unsigned long nruns = 10;

#ifdef USE_OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif

  unsigned long verbosity = 3;
  bool json = 0;
  std::string json_fname = "";

  while ((c = getopt_long(argc, argv, shortargs, longargs, nullptr)) != -1) {
    switch (c) {
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

    case 'd':
      try {
        delta = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Delta" << std::endl;
        return -1;
      }
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
      try {
        count = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Count" << std::endl;
        return -1;
      }
      break;

    case 'p':
      pattern_string << optarg;
      if (pattern_parser(pattern_string, pattern) != 0)
        return -1;
      break;

    case 'r':
      try {
        nruns = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Number of Runs" << std::endl;
        return -1;
      }
      break;

    case 's':
      pattern_scatter_string << optarg;
      if (pattern_parser(pattern_scatter_string, pattern_scatter) != 0)
        return -1;
      break;

    case 't':
      try {
        nthreads = std::stoi(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Number of Threads" << std::endl;
        return -1;
      }
      break;

    case 'v':
      try {
        verbosity = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Verbosity Level" << std::endl;
        return -1;
      }
      break;

    case 'w':
      try {
        wrap = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Wrap" << std::endl;
        return -1;
      }
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

  if (!json) {
    std::unique_ptr<Spatter::ConfigurationBase> c;
    if (backend.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(kernel,
          pattern, pattern_gather, pattern_scatter, delta, wrap, count, nruns,
          verbosity);
#ifdef USE_OPENMP
    else if (backend.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(kernel,
          pattern, pattern_gather, pattern_scatter, delta, wrap, count,
          nthreads, nruns, verbosity);
#endif
#ifdef USE_CUDA
    else if (backend.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(kernel,
          pattern, pattern_gather, pattern_scatter, delta, wrap, count, nruns,
          verbosity);
#endif
    else {
      std::cerr << "Invalid Backend " << backend << std::endl;
      return -1;
    }

    cl.configs.push_back(std::move(c));
  } else {
    Spatter::JSONParser json_file = Spatter::JSONParser(
        json_fname, backend, kernel, nthreads, nruns, verbosity);

    for (size_t i = 0; i < json_file.size(); ++i) {
      std::unique_ptr<Spatter::ConfigurationBase> c = json_file[i];
      cl.configs.push_back(std::move(c));
    }
  }

  return 0;
}

} // namespace Spatter

#endif
