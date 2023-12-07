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
      const std::string kernel = "gather", const int nthreads = 1,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : backend_(backend), default_kernel_(kernel),
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

      if (!v.contains("nthreads"))
        v["nthreads"] = default_omp_threads_;

      if (!v.contains("nruns"))
        v["nruns"] = default_nruns_;

      if (!v.contains("verbosity"))
        v["verbosity"] = default_verbosity_;
    }
  }

  size_t size() { return size_; }

  std::unique_ptr<Spatter::ConfigurationBase> operator[](size_t index) {
    assert(index < (size_));

    assert(data_[index].contains("kernel"));
    assert(data_[index].contains("pattern"));
    assert(data_[index].contains("nthreads"));
    assert(data_[index].contains("nruns"));
    assert(data_[index].contains("verbosity"));

    std::vector<size_t> pattern;
    std::string type;
    std::vector<std::vector<size_t>> generator;

    if (data_[index]["pattern"].type() == json::value_t::string) {
      std::string pattern_string =
          data_[index]["pattern"].template get<std::string>();
      pattern_string.erase(
          std::remove(pattern_string.begin(), pattern_string.end(), '\"'),
          pattern_string.end());

      std::stringstream pattern_stream;
      pattern_stream << pattern_string;

      if (pattern_parser(pattern_stream, pattern, type, generator) != 0)
        exit(1);
    } else {
      pattern = data_[index]["pattern"].template get<std::vector<size_t>>();
    }

    std::unique_ptr<Spatter::ConfigurationBase> c;

    if (backend_.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(
          data_[index]["kernel"], pattern, data_[index]["nruns"],
          data_[index]["verbosity"]);
#ifdef USE_OPENMP
    else if (backend_.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(
          data_[index]["kernel"], pattern, data_[index]["nthreads"],
          data_[index]["nruns"], data_[index]["verbosity"]);
#endif
#ifdef USE_CUDA
    else if (backend_.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(
          data_[index]["kernel"], pattern, data_[index]["nruns"],
          data_[index]["verbosity"]);
#endif
    else {
      std::cerr << "Invalid Backend " << backend_ << std::endl;
      exit(1);
    }

    return c;
  }

private:
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
  const int default_omp_threads_;
  const unsigned long default_nruns_;
  const unsigned long default_verbosity_;
};

} // namespace Spatter

#endif
