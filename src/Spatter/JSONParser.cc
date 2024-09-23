/*!
  \file JSONParser.cc
*/

#include "JSONParser.hh"

using json = nlohmann::json;

namespace Spatter {

JSONParser::JSONParser(std::string filename, aligned_vector<double> &sparse,
    double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const std::string backend, const bool aggregate,
    const bool atomic, const bool compress, size_t shared_mem,
    const int nthreads, const unsigned long verbosity, const std::string name,
    const std::string kernel, const size_t pattern_size, const size_t delta,
    const size_t delta_gather, const size_t delta_scatter,
    const size_t boundary, const long int seed, const size_t wrap,
    const size_t count, const size_t local_work_size, const unsigned long nruns)
    : sparse(sparse), dev_sparse(dev_sparse), sparse_size(sparse_size),
      sparse_gather(sparse_gather), dev_sparse_gather(dev_sparse_gather),
      sparse_gather_size(sparse_gather_size), sparse_scatter(sparse_scatter),
      dev_sparse_scatter(dev_sparse_scatter),
      sparse_scatter_size(sparse_scatter_size), dense(dense),
      dense_perthread(dense_perthread), dev_dense(dev_dense),
      dense_size(dense_size), backend_(backend), aggregate_(aggregate),
      atomic_(atomic), compress_(compress), shared_mem_(shared_mem),
      omp_threads_(nthreads), verbosity_(verbosity), default_name_(name),
      default_kernel_(kernel), default_pattern_size_(pattern_size),
      default_delta_(delta), default_delta_gather_(delta_gather),
      default_delta_scatter_(delta_scatter), default_boundary_(boundary),
      default_seed_(seed), default_wrap_(wrap), default_count_(count),
      default_local_work_size_(local_work_size), default_nruns_(nruns) {
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

    if (!v.contains("local-work-size") || (v["local-work-size"] <= -1))
      v["local-work-size"] = default_local_work_size_;

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

  assert(data_[index].contains("local-work-size"));
  assert(data_[index].contains("nruns"));

  aligned_vector<size_t> pattern;
  aligned_vector<size_t> pattern_gather;
  aligned_vector<size_t> pattern_scatter;

  size_t pattern_size = data_[index]["pattern-size"];
  size_t delta = data_[index]["delta"];
  size_t delta_gather = data_[index]["delta-gather"];
  size_t delta_scatter = data_[index]["delta-scatter"];
  size_t boundary = data_[index]["boundary"];

  if (data_[index].contains("pattern")) {
    if (get_pattern_("pattern", pattern, delta, index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern, boundary, this->size()) > boundary)
       exit(1);

    if (compress_)
      compress_pattern(pattern);
  }

  if (data_[index].contains("pattern-gather")) {
    if (get_pattern_("pattern-gather", pattern_gather, delta_gather,
        index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern_gather, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern_gather, boundary, this->size()) > boundary)
      exit(1);

    if (compress_)
      compress_pattern(pattern_gather);
  }

  if (data_[index].contains("pattern-scatter")) {
    if (get_pattern_("pattern-scatter", pattern_scatter, delta_scatter,
        index) != 0)
      exit(1);

    if (pattern_size > 0)
      if (truncate_pattern(pattern_scatter, pattern_size) != 0)
        exit(1);

    if (remap_pattern(pattern_scatter, boundary, this->size()) > boundary)
      exit(1);

    if (compress_)
      compress_pattern(pattern_scatter);
  }

  std::unique_ptr<Spatter::ConfigurationBase> c;
  if (backend_.compare("serial") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::Serial>>(index,
        data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
        pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
        dev_sparse_gather, sparse_gather_size, sparse_scatter,
        dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
        dev_dense, dense_size, delta, delta_gather, delta_scatter,
        data_[index]["seed"], data_[index]["wrap"], data_[index]["count"],
        data_[index]["nruns"], aggregate_, verbosity_);
#ifdef USE_OPENMP
  else if (backend_.compare("openmp") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(index,
        data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
        pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
        dev_sparse_gather, sparse_gather_size, sparse_scatter,
        dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
        dev_dense, dense_size, delta, delta_gather, delta_scatter,
        data_[index]["seed"], data_[index]["wrap"], data_[index]["count"],
        omp_threads_, data_[index]["nruns"], aggregate_, atomic_, verbosity_);
#endif
#ifdef USE_CUDA
  else if (backend_.compare("cuda") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(index,
        data_[index]["name"], data_[index]["kernel"], pattern, pattern_gather,
        pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
        dev_sparse_gather, sparse_gather_size, sparse_scatter,
        dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
        dev_dense, dense_size,delta, delta_gather, delta_scatter,
        data_[index]["seed"], data_[index]["wrap"], data_[index]["count"],
        shared_mem_, data_[index]["local-work-size"], data_[index]["nruns"],
        aggregate_, atomic_, verbosity_);
#endif
  else {
    std::cerr << "Invalid Backend " << backend_ << std::endl;
    exit(1);
  }

  return c;
}

int JSONParser::get_pattern_(const std::string &pattern_key,
    aligned_vector<size_t> &pattern, size_t &delta, const size_t index) {
  if (data_[index][pattern_key].type() == json::value_t::string) {
    std::string pattern_string =
        data_[index][pattern_key].template get<std::string>();
    pattern_string.erase(
        std::remove(pattern_string.begin(), pattern_string.end(), '\"'),
        pattern_string.end());

    std::stringstream pattern_stream;
    pattern_stream << pattern_string;

    return pattern_parser(pattern_stream, pattern, delta);
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
