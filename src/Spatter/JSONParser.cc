/*!
  \file JSONParser.cc
*/

#include <nlohmann/json.hpp>

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
    const bool atomic, const bool atomic_fence, const bool compress,
    const bool dense_buffers, size_t shared_mem, const int nthreads,
    const unsigned long verbosity, const std::string name,
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
      atomic_(atomic), atomic_fence_(atomic_fence), compress_(compress),
      dense_buffers_(dense_buffers), shared_mem_(shared_mem),
      omp_threads_(nthreads), verbosity_(verbosity),
      default_name_(name), default_kernel_(kernel),
      default_pattern_size_(pattern_size), default_delta_(delta),
      default_delta_gather_(delta_gather),
      default_delta_scatter_(delta_scatter), default_boundary_(boundary),
      default_seed_(seed), default_wrap_(wrap), default_count_(count),
      default_local_work_size_(local_work_size), default_nruns_(nruns) {
  if (!file_exists_(filename)) {
    std::cerr << "File does not exist" << std::endl;
    exit(1);
  }

  std::ifstream f(filename);
  std::function<void(void *)> json_deleter;
  json_deleter = [](void *json_ptr) { delete static_cast<json *>(json_ptr); };
  data_ = std::unique_ptr<void,
    std::function<void(void *)>>(new json(json::parse(f)), json_deleter);
  auto data_json_ptr = static_cast<json *>(data_.get());
  size_ = data_json_ptr->size();

  for (const auto &[key, v] : data_json_ptr->items()) {
    if (!v.contains("name"))
      v["name"] = default_name_;
    else if (!v["name"].is_string())
      throw std::invalid_argument("Invalid Name");

    if (!v.contains("kernel"))
      v["kernel"] = default_kernel_;
    else if (!v["kernel"].is_string())
      throw std::invalid_argument("Invalid Kernel Type");
    else {
      std::string kernel = v["kernel"].get<std::string>();
      std::transform(kernel.begin(), kernel.end(), kernel.begin(),
          [](unsigned char c) { return std::tolower(c); });

      if ((kernel.compare("gather") != 0) && (kernel.compare("scatter") != 0) &&
          (kernel.compare("gs") != 0) && (kernel.compare("multigather") != 0) &&
          (kernel.compare("multiscatter") != 0)) {
        throw std::invalid_argument("Invalid Kernel Type");
      }

      v["kernel"] = kernel;
    }

    if (v["kernel"].get<std::string>().compare("multigather") == 0) {
      if (!v.contains("pattern"))
        throw std::invalid_argument("Missing Pattern");
      else if (!v.contains("pattern-gather"))
        throw std::invalid_argument("Missing Pattern Gather");
    } else if (v["kernel"].get<std::string>().compare("multiscatter") == 0) {
      if (!v.contains("pattern"))
        throw std::invalid_argument("Missing Pattern");
      else if (!v.contains("pattern-scatter"))
        throw std::invalid_argument("Missing Pattern Scatter");
    } else if (v["kernel"].get<std::string>().compare("gs") == 0) {
      if (!v.contains("pattern-gather"))
        throw std::invalid_argument("Missing Pattern Gather");
      else if (!v.contains("pattern-scatter"))
        throw std::invalid_argument("Missing Pattern Scatter");
    } else {
      if (!v.contains("pattern"))
        throw std::invalid_argument("Missing Pattern");
    }

    if (!v.contains("pattern-size"))
      v["pattern-size"] = default_pattern_size_;
    else if (!v["pattern-size"].is_number())
      throw std::invalid_argument("Invalid Pattern Size");
    else if (v["pattern-size"].get<int64_t>() < 1)
      throw std::invalid_argument("Invalid Pattern Size");

    if (!v.contains("delta"))
      v["delta"] = default_delta_;
    else if (!v["delta"].is_number())
      throw std::invalid_argument("Invalid Delta");
    else if (v["delta"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Delta");

    if (!v.contains("delta-gather"))
      v["delta-gather"] = default_delta_gather_;
    else if (!v["delta-gather"].is_number())
      throw std::invalid_argument("Invalid Delta Gather");
    else if (v["delta-gather"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Delta Gather");

    if (!v.contains("delta-scatter"))
      v["delta-scatter"] = default_delta_scatter_;
    else if (!v["delta-scatter"].is_number())
      throw std::invalid_argument("Invalid Delta Scatter");
    else if (v["delta-scatter"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Delta Scatter");

    if (!v.contains("boundary"))
      v["boundary"] = default_boundary_;
    else if (!v["boundary"].is_number())
      throw std::invalid_argument("Invalid Boundary");
    else if (v["boundary"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Boundary");

    if (!v.contains("seed"))
      v["seed"] = default_seed_;
    else if (!v["seed"].is_number())
      throw std::invalid_argument("Invalid Random Seed");
    else if (v["seed"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Random Seed");

    if (!v.contains("wrap"))
      v["wrap"] = default_wrap_;
    else if (!v["wrap"].is_number())
      throw std::invalid_argument("Invalid Wrap");
    else if (v["wrap"].get<int64_t>() < 1)
      throw std::invalid_argument("Invalid Wrap");

    if (!v.contains("count"))
      v["count"] = default_count_;
    else if (!v["count"].is_number())
      throw std::invalid_argument("Invalid Count");
    else if (v["count"].get<int64_t>() < 1)
      throw std::invalid_argument("Invalid Count");

    if (!v.contains("local-work-size"))
      v["local-work-size"] = default_local_work_size_;
    else if (!v["local-work-size"].is_number())
      throw std::invalid_argument("Invalid Local Work Size");
    else if (v["local-work-size"].get<int64_t>() < 0)
      throw std::invalid_argument("Invalid Local Work Size");

    if (!v.contains("nruns"))
      v["nruns"] = default_nruns_;
    else if (!v["nruns"].is_number())
      throw std::invalid_argument("Invalid Number of Runs");
    else if (v["nruns"].get<int64_t>() < 1)
      throw std::invalid_argument("Invalid Number of Runs");
  }
}

size_t JSONParser::size() { return size_; }

std::unique_ptr<Spatter::ConfigurationBase> JSONParser::operator[](
    const size_t index) {
  auto data_json_ptr = static_cast<json *>(data_.get());

  assert(index < (size_));

  assert((*data_json_ptr)[index].contains("name"));

  assert((*data_json_ptr)[index].contains("kernel"));
  assert((*data_json_ptr)[index]["kernel"].type() == json::value_t::string);

  assert((*data_json_ptr)[index].contains("pattern-size"));
  assert((*data_json_ptr)[index].contains("delta"));
  assert((*data_json_ptr)[index].contains("delta-gather"));
  assert((*data_json_ptr)[index].contains("delta-scatter"));
  assert((*data_json_ptr)[index].contains("boundary"));
  assert((*data_json_ptr)[index].contains("seed"));
  assert((*data_json_ptr)[index].contains("wrap"));
  assert((*data_json_ptr)[index].contains("count"));

  assert((*data_json_ptr)[index].contains("local-work-size"));
  assert((*data_json_ptr)[index].contains("nruns"));

  aligned_vector<size_t> pattern;
  aligned_vector<size_t> pattern_gather;
  aligned_vector<size_t> pattern_scatter;

  size_t pattern_size = (*data_json_ptr)[index]["pattern-size"];
  size_t delta = (*data_json_ptr)[index]["delta"];
  size_t delta_gather = (*data_json_ptr)[index]["delta-gather"];
  size_t delta_scatter = (*data_json_ptr)[index]["delta-scatter"];
  size_t boundary = (*data_json_ptr)[index]["boundary"];

  if ((*data_json_ptr)[index].contains("pattern")) {
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

  if ((*data_json_ptr)[index].contains("pattern-gather")) {
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

  if ((*data_json_ptr)[index].contains("pattern-scatter")) {
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
        (*data_json_ptr)[index]["name"], (*data_json_ptr)[index]["kernel"],
        pattern, pattern_gather, pattern_scatter, sparse, dev_sparse,
        sparse_size, sparse_gather, dev_sparse_gather, sparse_gather_size,
        sparse_scatter, dev_sparse_scatter, sparse_scatter_size, dense,
        dense_perthread, dev_dense, dense_size, delta, delta_gather,
        delta_scatter, (*data_json_ptr)[index]["seed"],
        (*data_json_ptr)[index]["wrap"], (*data_json_ptr)[index]["count"],
        (*data_json_ptr)[index]["nruns"], aggregate_, verbosity_);
#ifdef USE_OPENMP
  else if (backend_.compare("openmp") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::OpenMP>>(index,
        (*data_json_ptr)[index]["name"], (*data_json_ptr)[index]["kernel"],
        pattern, pattern_gather, pattern_scatter, sparse, dev_sparse,
        sparse_size, sparse_gather, dev_sparse_gather, sparse_gather_size,
        sparse_scatter, dev_sparse_scatter, sparse_scatter_size, dense,
        dense_perthread, dev_dense, dense_size, delta, delta_gather,
        delta_scatter, (*data_json_ptr)[index]["seed"],
        (*data_json_ptr)[index]["wrap"], (*data_json_ptr)[index]["count"],
        omp_threads_, (*data_json_ptr)[index]["nruns"], aggregate_, atomic_,
        atomic_fence_, dense_buffers_, verbosity_);
#endif
#ifdef USE_CUDA
  else if (backend_.compare("cuda") == 0)
    c = std::make_unique<Spatter::Configuration<Spatter::CUDA>>(index,
        (*data_json_ptr)[index]["name"], (*data_json_ptr)[index]["kernel"],
        pattern, pattern_gather, pattern_scatter, sparse, dev_sparse,
        sparse_size, sparse_gather, dev_sparse_gather, sparse_gather_size,
        sparse_scatter, dev_sparse_scatter, sparse_scatter_size, dense,
        dense_perthread, dev_dense, dense_size,delta, delta_gather,
        delta_scatter, (*data_json_ptr)[index]["seed"],
        (*data_json_ptr)[index]["wrap"], (*data_json_ptr)[index]["count"],
        shared_mem_, (*data_json_ptr)[index]["local-work-size"],
        (*data_json_ptr)[index]["nruns"], aggregate_, atomic_, verbosity_);
#endif
  else {
    std::cerr << "Invalid Backend " << backend_ << std::endl;
    exit(1);
  }

  return c;
}

int JSONParser::get_pattern_(const std::string &pattern_key,
    aligned_vector<size_t> &pattern, size_t &delta, const size_t index) {
  auto data_json_ptr = static_cast<json *>(data_.get());
  if ((*data_json_ptr)[index][pattern_key].type() == json::value_t::string) {
    std::string pattern_string =
        (*data_json_ptr)[index][pattern_key].template get<std::string>();
    pattern_string.erase(
        std::remove(pattern_string.begin(), pattern_string.end(), '\"'),
        pattern_string.end());

    std::stringstream pattern_stream;
    pattern_stream << pattern_string;

    return pattern_parser(pattern_stream, pattern, delta);
  } else {
    pattern = (*data_json_ptr)[index][pattern_key].template get<aligned_vector<size_t>>();
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
