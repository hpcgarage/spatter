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

#include "Spatter/Configuration.hh"

using json = nlohmann::json;

namespace Spatter {

class JSONParser {
public:
  JSONParser(std::string filename, std::string backend = "serial",
      std::string kernel = "gather", unsigned long nruns = 10,
      unsigned long verbosity = 3)
      : backend_(backend), default_kernel_(kernel), default_nruns_(nruns),
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
    assert(data_[index].contains("nruns"));
    assert(data_[index].contains("verbosity"));

    std::unique_ptr<Spatter::ConfigurationBase> c;

    if (backend_.compare("serial") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(
          data_[index]["kernel"], data_[index]["pattern"],
          data_[index]["nruns"], data_[index]["verbosity"]);
#ifdef USE_OPENMP
    else if (backend_.compare("openmp") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(
          data_[index]["kernel"], data_[index]["pattern"],
          data_[index]["nruns"], data_[index]["verbosity"]);
#endif
#ifdef USE_CUDA
    else if (backend_.compare("cuda") == 0)
      c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(
          data_[index]["kernel"], data_[index]["pattern"],
          data_[index]["nruns"], data_[index]["verbosity"]);
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

  std::string default_kernel_;
  unsigned long default_nruns_;
  unsigned long default_verbosity_;
};

} // namespace Spatter

#endif
