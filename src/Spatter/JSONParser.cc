/*!
  \file JSONParser.cc
*/

#include "JSONParser.hh"

using json = nlohmann::json;

namespace Spatter {

JSONParser::JSONParser(std::string filename, const std::string backend,
    const bool aggregate, const bool atomic, const bool compress,
    const unsigned long verbosity, const std::string name,
    const std::string kernel, const size_t pattern_size, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter,
    const size_t boundary, const int seed, const size_t wrap,
    const size_t count, const int nthreads, const unsigned long nruns)
    : backend_(backend), aggregate_(aggregate), atomic_(atomic),
      compress_(compress), verbosity_(verbosity), default_name_(name),
      default_kernel_(kernel), default_pattern_size_(pattern_size),
      default_delta_(delta), default_delta_gather_(delta_gather),
      default_delta_scatter_(delta_scatter), default_boundary_(boundary),
      default_seed_(seed), default_wrap_(wrap), default_count_(count),
      default_omp_threads_(nthreads), default_nruns_(nruns) {
  if (!file_exists_(filename)) {
    std::cerr << "File does not exist" << std::endl;
    exit(1);
  }

  std::ifstream f(filename);
  data_ = json::parse(f);
  size_ = data_.size();

  for (const auto &[key, v] : data_.items()) {
    if (!v.contains("name"))
      v["name"] = default_name_;

    if (!v.contains("kernel")) {
      v["kernel"] = default_kernel_;

      assert(v.contains("pattern"));
    } else {
      std::string kernel = v["kernel"];
      std::transform(kernel.begin(), kernel.end(), kernel.begin(),
          [](unsigned char c) { return std::tolower(c); });

      // The kernel may be specified as 'GS' instead of 'sg'
      kernel = (kernel.compare("gs") == 0) ? "sg" : kernel;
      v["kernel"] = kernel;

      if (kernel.compare("sg") == 0) {
        // This kernel does not require --pattern to be specified
        assert(v.contains("pattern-gather") && v.contains("pattern-scatter"));
      } else {
        assert(v.contains("pattern"));
      }
    }

    if (!v.contains("pattern-size") || (v["pattern-size"] <= -1))
      v["pattern-size"] = default_pattern_size_;

    if (!v.contains("delta") || (v["delta"] <= -1))
      v["delta"] = default_delta_;

    if (!v.contains("delta-gather") || (v["delta-gather"] <= -1))
      v["delta-gather"] = default_delta_gather_;

    if (!v.contains("delta-scatter") || (v["delta-scatter"] <= -1))
      v["delta-scatter"] = default_delta_scatter_;

    if (!v.contains("boundary"))
      v["boundary"] = default_boundary_;

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

size_t JSONParser::size() { return size_; }

std::unique_ptr<Spatter::ConfigurationBase> JSONParser::operator[](
    const size_t index) {
  assert(index < (size_));

  assert(data_[index].contains("name"));

  assert(data_[index].contains("kernel"));
  assert(data_[index]["kernel"].type() == json::value_t::string);

  assert(data_[index].contains("pattern-size"));
  assert(data_[index].contains("delta"));
  assert(data_[index].contains("delta-gather"));
  assert(data_[index].contains("delta-scatter"));
  assert(data_[index].contains("boundary"));
  assert(data_[index].contains("seed"));
  assert(data_[index].contains("wrap"));
  assert(data_[index].contains("count"));

  assert(data_[index].contains("nthreads"));
  assert(data_[index].contains("nruns"));

  aligned_vector<size_t> pattern;
  aligned_vector<size_t> pattern_gather;
  aligned_vector<size_t> pattern_scatter;

  size_t pattern_size = data_[index]["pattern-size"];
  size_t boundary = data_[index]["boundary"];

  if (data_[index].contains("pattern")) {
    if (get_pattern_("pattern", pattern, index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern, boundary) > boundary)
       exit(1);
  }

  if (data_[index].contains("pattern-gather")) {
    if (get_pattern_("pattern-gather", pattern_gather, index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern_gather, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern_gather, boundary) > boundary)
      exit(1);
  }

  if (data_[index].contains("pattern-scatter")) {
    if (get_pattern_("pattern-scatter", pattern_scatter, index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern_scatter, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern_scatter, boundary) > boundary)
      exit(1);
  }

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
        data_[index]["name"], data_[index]["kernel"], pattern, pattern_scatter,
        pattern_gather, data_[index]["delta"], data_[index]["delta-gather"],
        data_[index]["delta-scatter"], data_[index]["seed"],
        data_[index]["wrap"], data_[index]["count"], data_[index]["nthreads"],
        data_[index]["nruns"], aggregate_, atomic_, compress_, verbosity_);
#endif
#ifdef USE_CUDA
  else if (backend_.compare("cuda") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(index,
        data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
        pattern_scatter, data_[index]["delta"], data_[index]["delta-gather"],
        data_[index]["delta-scatter"], data_[index]["seed"],
        data_[index]["wrap"], data_[index]["count"], data_[index]["nruns"],
        aggregate_, atomic_, compress_, verbosity_);
#endif
  else {
    std::cerr << "Invalid Backend " << backend_ << std::endl;
    exit(1);
  }

  return c;
}

int JSONParser::get_pattern_(const std::string &pattern_key,
    aligned_vector<size_t> &pattern, const size_t index) {
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

bool JSONParser::file_exists_(const std::string &fpth) {
  bool exists_ = false;
  if (FILE *file = fopen(fpth.c_str(), "r")) {
    fclose(file);
    exists_ = true;
  }

  return exists_;
}
} // namespace Spatter
