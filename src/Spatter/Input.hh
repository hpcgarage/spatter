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

#include "Spatter/Configuration.hh"
#include "Spatter/JSONParser.hh"
#include "Spatter/SpatterTypes.hh"

namespace Spatter {
static char *shortargs = (char *)"b:f:hk:n:p:v:";

struct ClArgs {
  std::vector<std::unique_ptr<Spatter::ConfigurationBase>> configs;
};

void help(char *progname) {
  std::cout << "Spatter\n";
  std::cout << "Usage: " << progname << "\n";
  std::cout << std::left << std::setw(10) << "-b" << std::setw(40)
            << "Backend (default serial)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-f" << std::setw(40)
            << "Input File" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-h" << std::setw(40)
            << "Print Help Message" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-k" << std::setw(40)
            << "Kernel (default gather)" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-n" << std::setw(40)
            << "Set Number of Runs" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-p" << std::setw(40)
            << "Set Pattern" << std::left << "\n";
  std::cout << std::left << std::setw(10) << "-v" << std::setw(40)
            << "Set Verbosity Level" << std::left << "\n";
}

void usage(char *progname) {
  std::cout << "Usage: " << progname
            << "[-b backend] [-f input file] [-h help] [-k kernel] [-n nruns] "
               "[-p pattern] [-v "
               "verbosity]"
            << std::endl;
}

int generate_pattern(std::string type, std::vector<size_t> generator,
    std::vector<size_t> &pattern) {
  if (type.compare("UNIFORM") == 0) {
    size_t len = generator[0];
    size_t stride = generator[1];

    for (size_t i = 0; i < len; ++i)
      pattern.push_back(i * stride);
  } else {
    std::cerr << "Parsing Error: Invalid Pattern Generator Type (Valid types "
                 "are: UNIFORM)"
              << std::endl;
    return -1;
  }

  return 0;
}

int parse_input(const int argc, char **argv, ClArgs &cl) {
  int c;
  std::stringstream pattern_string;
  std::vector<size_t> pattern;
  std::vector<size_t> generator;

  std::string backend = "serial";
  std::string kernel = "gather";
  std::string type = "";
  unsigned long nruns = 10;
  unsigned long verbosity = 3;
  bool json = 0;

  while ((c = getopt(argc, argv, shortargs)) != -1) {
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

    case 'f': {
      json = 1;
      Spatter::JSONParser json_file = Spatter::JSONParser(optarg);

      for (size_t i = 0; i < json_file.size(); ++i) {
        std::unique_ptr<Spatter::ConfigurationBase> c = json_file[i];
        cl.configs.push_back(std::move(c));
      }
      break;
    }

    case 'h':
      help(argv[0]);
      return -1;

    case 'k':
      kernel = optarg;
      if ((kernel.compare("gather") != 0) && (kernel.compare("scatter") != 0)) {
        std::cerr << "Valid Kernels are: gather, scatter" << std::endl;
        return -1;
      }
      break;

    case 'n':
      try {
        nruns = std::stoul(optarg);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Number of Runs" << std::endl;
        return -1;
      }
      break;

    case 'p':
      pattern_string << optarg;
      if (pattern_string.str().rfind("UNIFORM", 0) == 0) {
        std::getline(pattern_string, type, ':');

        for (std::string line; std::getline(pattern_string, line, ':');) {
          try {
            size_t val = std::stoul(line);

            if (line[0] == '-') {
              std::cerr
                  << "Parsing Error: Found Negative Index in Pattern Generator"
                  << std::endl;
              return -1;
            } else
              generator.push_back(val);
          } catch (const std::invalid_argument &ia) {
            std::cerr << "Parsing Error: Invalid Pattern Generator Format"
                      << std::endl;
            return -1;
          }
        }
      }

      if (!type.empty())
        if (generate_pattern(type, generator, pattern) != 0)
          return -1;

      if (type.empty()) {
        for (std::string line; std::getline(pattern_string, line, ',');) {
          try {
            size_t val = std::stoul(line);

            if (line[0] == '-') {
              std::cerr << "Parsing Error: Found Negative Index in Pattern"
                        << std::endl;
              return -1;
            } else
              pattern.push_back(val);
          } catch (const std::invalid_argument &ia) {
            std::cerr << "Parsing Error: Invalid Pattern Format" << std::endl;
            return -1;
          }
        }
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

    case '?':
      usage(argv[0]);
      return -1;
    }
  }

  if (!json) {
    std::unique_ptr<Spatter::ConfigurationBase> c;
    if (backend.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(
          kernel, pattern, nruns, verbosity);
#ifdef USE_OPENMP
    else if (backend.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(
          kernel, pattern, nruns, verbosity);
#endif
#ifdef USE_CUDA
    else if (backend.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(
          kernel, pattern, nruns, verbosity);
#endif
    else {
      std::cerr << "Invalid Backend " << backend << std::endl;
      return -1;
    }

    cl.configs.push_back(std::move(c));
  }

  return 0;
}

} // namespace Spatter

#endif
