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
  JSONParser(std::string filename, aligned_vector<double> &sparse,
      double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const std::string backend,
      const bool aggregate, const bool atomic, const bool compress,
      const size_t shared_mem, const int nthreads,
      const unsigned long verbosity, const std::string name = "",
      const std::string kernel = "gather", const size_t pattern_size = 0,
      const size_t delta = 8, const size_t delta_gather = 8,
      const size_t delta_scatter = 8, const size_t boundary = 0,
      const long int seed = -1, const size_t wrap = 1,
      const size_t count = 1024, const size_t local_work_size = 1024,
      const unsigned long nruns = 10);

  size_t size();

  std::unique_ptr<Spatter::ConfigurationBase> operator[](const size_t index);

private:
  int get_pattern_(const std::string &pattern_key,
      aligned_vector<size_t> &pattern, size_t &delta, const size_t index);
  bool file_exists_(const std::string &fpth);

private:
  json data_;
  size_t size_;

  aligned_vector<double> &sparse;
  double *&dev_sparse;
  size_t &sparse_size;

  aligned_vector<double> &sparse_gather;
  double *&dev_sparse_gather;
  size_t &sparse_gather_size;

  aligned_vector<double> &sparse_scatter;
  double *&dev_sparse_scatter;
  size_t &sparse_scatter_size;

  aligned_vector<double> &dense;
  aligned_vector<aligned_vector<double>> &dense_perthread;
  double *&dev_dense;
  size_t &dense_size;

  std::string backend_;
  const bool aggregate_;
  const bool atomic_;
  const bool compress_;
  const size_t shared_mem_;
  const int omp_threads_;
  const unsigned long verbosity_;

  std::string default_name_;
  const std::string default_kernel_;
  const size_t default_pattern_size_;

  const size_t default_delta_;
  const size_t default_delta_gather_;
  const size_t default_delta_scatter_;

  const size_t default_boundary_;
  const long int default_seed_;
  const size_t default_wrap_;
  const size_t default_count_;

  const size_t default_local_work_size_;
  const unsigned long default_nruns_;
};

} // namespace Spatter

#endif
