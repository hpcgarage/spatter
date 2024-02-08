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
  JSONParser(std::string filename, const std::string backend,
      const bool aggregate, const bool compress, const unsigned long verbosity,
      const std::string name = "", const std::string kernel = "gather",
      const size_t delta = 8, const size_t delta_gather = 8,
      const size_t delta_scatter = 8, const int seed = -1,
      const size_t wrap = 1, const size_t count = 1024, const int nthreads = 1,
      const unsigned long nruns = 10)
      : backend_(backend), aggregate_(aggregate), compress_(compress),
        verbosity_(verbosity), default_name_(name), default_kernel_(kernel),
        default_delta_(delta), default_delta_gather_(delta_gather),
        default_delta_scatter_(delta_scatter), default_seed_(seed),
        default_wrap_(wrap), default_count_(count),
        default_omp_threads_(nthreads), default_nruns_(nruns) {
    if (!file_exists_(filename)) {
      std::cerr << "File does not exist" << std::endl;
      exit(1);
    }

    std::ifstream f(filename);
    data_ = json::parse(f);
    size_ = data_.size();

    for (const auto &[key, v] : data_.items()) {
      assert(v.contains("pattern"));

      if (!v.contains("name"))
        v["name"] = default_name_;

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

      if (!v.contains("delta-gather"))
        v["delta-gather"] = default_delta_gather_;

      if (!v.contains("delta-scatter"))
        v["delta-scatter"] = default_delta_scatter_;

      if (!v.contains("seed"))
        v["seed"] = default_seed_;

      if (!v.contains("wrap"))
        v["wrap"] = default_wrap_;

      if (!v.contains("count"))
        v["count"] = default_count_;

      if (!v.contains("nthreads"))
        v["nthreads"] = default_omp_threads_;

      if (!v.contains("nruns"))
        v["nruns"] = default_nruns_;
    }
  }

  size_t size() { return size_; }

  std::unique_ptr<Spatter::ConfigurationBase> operator[](const size_t index) {
    assert(index < (size_));

    assert(data_[index].contains("name"));

    assert(data_[index].contains("kernel"));
    assert(data_[index]["kernel"].type() == json::value_t::string);

    assert(data_[index].contains("delta"));
    assert(data_[index].contains("delta-gather"));
    assert(data_[index].contains("delta-scatter"));
    assert(data_[index].contains("seed"));
    assert(data_[index].contains("wrap"));
    assert(data_[index].contains("count"));

    assert(data_[index].contains("nthreads"));
    assert(data_[index].contains("nruns"));

    aligned_vector<size_t> pattern;
    aligned_vector<size_t> pattern_gather;
    aligned_vector<size_t> pattern_scatter;

    if (data_[index].contains("pattern"))
      if (get_pattern_("pattern", pattern, index) != 0)
        exit(1);

    if (data_[index].contains("pattern-gather"))
      if (get_pattern_("pattern-gather", pattern_gather, index) != 0)
        exit(1);

    if (data_[index].contains("pattern-scatter"))
      if (get_pattern_("pattern-scatter", pattern_scatter, index) != 0)
        exit(1);

    std::unique_ptr<Spatter::ConfigurationBase> c;
    if (backend_.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(index,
          data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
          pattern_scatter, data_[index]["delta"], data_[index]["delta-gather"],
          data_[index]["delta-scatter"], data_[index]["seed"],
          data_[index]["wrap"], data_[index]["count"], data_[index]["nruns"],
          aggregate_, compress_, verbosity_);
#ifdef USE_OPENMP
    else if (backend_.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(index,
          data_[index]["name"], data_[index]["kernel"], pattern,
          pattern_scatter, pattern_gather, data_[index]["delta"],
          data_[index]["delta-gather"], data_[index]["delta-scatter"],
          data_[index]["seed"], data_[index]["wrap"], data_[index]["count"],
          data_[index]["nthreads"], data_[index]["nruns"], aggregate_,
          compress_, verbosity_);
#endif
#ifdef USE_CUDA
    else if (backend_.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(index,
          data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
          pattern_scatter, data_[index]["delta"], data_[index]["delta-gather"],
          data_[index]["delta-scatter"], data_[index]["seed"],
          data_[index]["wrap"], data_[index]["count"], data_[index]["nruns"],
          aggregate_, compress_, verbosity_);
#endif
    else {
      std::cerr << "Invalid Backend " << backend_ << std::endl;
      exit(1);
    }

    return c;
  }

private:
  int get_pattern_(const std::string &pattern_key, aligned_vector<size_t> &pattern,
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
      pattern = data_[index][pattern_key].template get<aligned_vector<size_t>>();
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
  const bool aggregate_;
  const bool compress_;
  const unsigned long verbosity_;

  std::string default_name_;
  const std::string default_kernel_;

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
