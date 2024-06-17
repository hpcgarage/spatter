/*!
  \file JSONParser.hh
*/

#ifndef SPATTER_JSONPARSER_HH
#define SPATTER_JSONPARSER_HH

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "Configuration.hh"
#include "PatternParser.hh"

using json = nlohmann::json;

namespace Spatter {

class JSONParser {
public:
  JSONParser(std::string filename, const std::string backend,
      const bool aggregate, const bool atomic, const bool compress,
      const unsigned long verbosity, const std::string name = "",
      const std::string kernel = "gather", const size_t pattern_size = 0,
      const size_t delta = 8, const size_t delta_gather = 8,
      const size_t delta_scatter = 8, const int seed = -1,
      const size_t wrap = 1, const size_t count = 1024, const int nthreads = 1,
      const unsigned long nruns = 10);

  size_t size();

  std::unique_ptr<Spatter::ConfigurationBase> operator[](const size_t index);

private:
  int get_pattern_(const std::string &pattern_key,
      aligned_vector<size_t> &pattern, const size_t index);
  bool file_exists_(const std::string &fpth);

private:
  json data_;
  size_t size_;

  std::string backend_;
  const bool aggregate_;
  const bool atomic_;
  const bool compress_;
  const unsigned long verbosity_;

  std::string default_name_;
  const std::string default_kernel_;
  const size_t default_pattern_size_;

  const size_t default_delta_;
  const size_t default_delta_gather_;
  const size_t default_delta_scatter_;

  const int default_seed_;
  const size_t default_wrap_;
  const size_t default_count_;

  const int default_omp_threads_;
  const unsigned long default_nruns_;
};

} // namespace Spatter

#endif
