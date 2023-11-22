/*!
  \file Configuration.hh
*/

#ifndef SPATTER_CONFIGURATION_HH
#define SPATTER_CONFIGURATION_HH

#include <experimental/iterator>
#include <ostream>
#include <sstream>

#include "Spatter/SpatterTypes.hh"
#include "Spatter/Timer.hh"

namespace Spatter {

class ConfigurationBase {
public:
  ConfigurationBase(const std::string kernel, const std::vector<size_t> pattern,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : kernel(kernel), pattern(pattern), nruns(nruns), verbosity(verbosity) {}

  ~ConfigurationBase() = default;

  virtual void setup() = 0;

  virtual int run(bool timed) = 0;
  virtual void gather(bool timed) = 0;
  virtual void scatter(bool timed) = 0;

  virtual void report() = 0;

public:
  std::string kernel;
  const std::vector<size_t> pattern;
  std::vector<double> sparse;
  std::vector<double> dense;

  const unsigned long nruns;
  const unsigned long verbosity;

  Spatter::Timer timer;
};

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  if (config.verbosity >= 1)
    config_output << "Kernel: " << config.kernel;

  if (config.verbosity >= 2) {
    config_output << "\nPattern: ";
    std::copy(std::begin(config.pattern), std::end(config.pattern),
        std::experimental::make_ostream_joiner(config_output, ", "));
  }
  return out << config_output.str() << std::endl;
}

template <typename Backend> class Configuration : public ConfigurationBase {};

template <> class Configuration<Spatter::Serial> : public ConfigurationBase {
public:
  Configuration(const std::string kernel, const std::vector<size_t> pattern,
      const unsigned long nruns = 10, const unsigned long verbosity = 3)
      : ConfigurationBase(kernel, pattern, nruns, verbosity){};

  void setup() {
    if (verbosity >= 3)
      std::cout << "Spatter Serial Setup" << std::endl;

    dense.resize(pattern.size());

    size_t max_pattern_val = 0;
    for (size_t i = 0; i < pattern.size(); ++i)
      if (pattern[i] > max_pattern_val)
        max_pattern_val = pattern[i];

    sparse.resize(max_pattern_val);

    for (size_t i = 0; i < max_pattern_val; ++i)
      sparse[i] = rand();

    if (verbosity >= 3)
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDense Array Size: " << dense.size()
                << "\tSparse Array Size: " << sparse.size()
                << "\tMax Pattern Val: " << max_pattern_val << std::endl;
  }

  int run(bool timed) {
    if (kernel.compare("gather") == 0)
      gather(timed);
    else if (kernel.compare("scatter") == 0)
      scatter(timed);
    else {
      std::cerr << "Invalid Kernel Type" << std::endl;
      return -1;
    }

    return 0;
  }

  void gather(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Gather Serial Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      dense[i] = sparse[pattern[i]];

    if (timed)
      timer.stop();
  }

  void scatter(bool timed) {
    if (verbosity >= 3)
      std::cout << "Spatter Scatter Serial Running" << std::endl;

    if (timed)
      timer.start();

    for (size_t i = 0; i < pattern.size(); ++i)
      sparse[pattern[i]] = dense[i];

    if (timed)
      timer.stop();
  }

  void report() {
    std::cout << "Spatter Serial Report" << std::endl;
    std::cout << nruns * pattern.size() * sizeof(size_t) << " Total Bytes Moved"
              << std::endl;
    std::cout << pattern.size() * sizeof(size_t) << " Bytes Moved per Run"
              << std::endl;
    std::cout << nruns << " Runs took " << std::fixed << timer.seconds()
              << " Seconds" << std::endl;
    std::cout << "Average Bandwidth: "
              << (double)(nruns * pattern.size() * sizeof(size_t)) /
            timer.seconds() / 1000000.
              << " MB/s" << std::endl;
  }
};

} // namespace Spatter

#endif
