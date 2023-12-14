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

#include "Spatter/Configuration.hh"
#include "Spatter/PatternParser.hh"

using json = nlohmann::json;

namespace Spatter {

class JSONParser {
public:
  JSONParser(std::string filename, const std::string backend = "serial",
      const std::string kernel = "gather", const size_t delta = 8,
      const size_t wrap = 1, const size_t count = 1024, const int nthreads = 1,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : backend_(backend), default_kernel_(kernel), default_delta_(delta),
        default_wrap_(wrap), default_count_(count),
        default_omp_threads_(nthreads), default_nruns_(nruns),
        default_verbosity_(verbosity) {
    if (!file_exists_(filename)) {
      std::cerr << "File does not exist" << std::endl;
      exit(1);
    }

    std::ifstream f(filename);
    data_ = json::parse(f);
    size_ = data_.size();

    for (const auto &[key, v] : data_.items()) {
      assert(v.contains("pattern"));

      if (!v.contains("kernel"))
        v["kernel"] = default_kernel_;
      else {
        std::string kernel = v["kernel"];
        std::transform(kernel.begin(), kernel.end(), kernel.begin(),
            [](unsigned char c) { return std::tolower(c); });

        v["kernel"] = kernel;
      }

      if (!v.contains("delta"))
        v["delta"] = default_delta_;

      if (!v.contains("wrap"))
        v["wrap"] = default_wrap_;

      if (!v.contains("count"))
        v["count"] = default_count_;

      if (!v.contains("nthreads"))
        v["nthreads"] = default_omp_threads_;

      if (!v.contains("nruns"))
        v["nruns"] = default_nruns_;

      if (!v.contains("verbosity"))
        v["verbosity"] = default_verbosity_;
    }
  }

  size_t size() { return size_; }

  std::unique_ptr<Spatter::ConfigurationBase> operator[](const size_t index) {
    assert(index < (size_));

    assert(data_[index].contains("kernel"));
    assert(data_[index]["kernel"].type() == json::value_t::string);

    assert(data_[index].contains("delta"));
    assert(data_[index].contains("wrap"));
    assert(data_[index].contains("count"));

    assert(data_[index].contains("nthreads"));
    assert(data_[index].contains("nruns"));
    assert(data_[index].contains("verbosity"));

    std::vector<size_t> pattern;
    std::vector<size_t> pattern_gather;
    std::vector<size_t> pattern_scatter;

    if (get_pattern_("pattern", pattern, index) != 0)
      exit(1);

    if (get_pattern_("pattern-gather", pattern_gather, index) != 0)
      exit(1);

    if (get_pattern_("pattern-scatter", pattern_scatter, index) != 0)
      exit(1);

    std::unique_ptr<Spatter::ConfigurationBase> c;

    if (backend_.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(
          data_[index]["kernel"], pattern, pattern_gather, pattern_scatter,
          data_[index]["delta"], data_[index]["wrap"], data_[index]["count"],
          data_[index]["nruns"], data_[index]["verbosity"]);
#ifdef USE_OPENMP
    else if (backend_.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(
          data_[index]["kernel"], pattern, pattern_scatter, pattern_gather,
          data_[index]["delta"], data_[index]["wrap"], data_[index]["count"],
          data_[index]["nthreads"], data_[index]["nruns"],
          data_[index]["verbosity"]);
#endif
#ifdef USE_CUDA
    else if (backend_.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(
          data_[index]["kernel"], pattern, pattern_gather, pattern_scatter,
          data_[index]["delta"], data_[index]["wrap"], data_[index]["count"],
          data_[index]["nruns"], data_[index]["verbosity"]);
#endif
    else {
      std::cerr << "Invalid Backend " << backend_ << std::endl;
      exit(1);
    }

    return c;
  }

private:
  int get_pattern_(const std::string &pattern_key, std::vector<size_t> &pattern,
      const size_t index) {
    if (data_[index][pattern_key].type() == json::value_t::string) {
      std::string pattern_string =
          data_[index][pattern_key].template get<std::string>();
      pattern_string.erase(
          std::remove(pattern_string.begin(), pattern_string.end(), '\"'),
          pattern_string.end());

      std::stringstream pattern_stream;
      pattern_stream << pattern_string;

      return pattern_parser(pattern_stream, pattern);
    } else {
      pattern = data_[index][pattern_key].template get<std::vector<size_t>>();
      return 0;
    }
  }

  bool file_exists_(const std::string &fpth) {
    bool exists_ = false;
    if (FILE *file = fopen(fpth.c_str(), "r")) {
      fclose(file);
      exists_ = true;
    }

    return exists_;
  }

private:
  json data_;
  size_t size_;
  std::string backend_;

  const std::string default_kernel_;

  const size_t default_delta_;
  const size_t default_wrap_;
  const size_t default_count_;

  const int default_omp_threads_;
  const unsigned long default_nruns_;
  const unsigned long default_verbosity_;
};

} // namespace Spatter

#endif
