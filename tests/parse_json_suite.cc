#include <iostream>
#include <string>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int json_test(size_t run_count, std::string kernel,
    std::vector<size_t> pattern_lengths,
    std::vector<std::vector<size_t>> patterns, std::vector<size_t> counts,
    int argc, char **argv) {
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != run_count) {
    std::cerr << "Test failure on JSON Parse: Expected number of runs to "
                 "be "
              << run_count << ", actually was " << cl.configs.size()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr << "Test failure on JSON Parse: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < run_count; ++i) {
    if (cl.configs[i]->pattern.size() != pattern_lengths[i]) {
      std::cerr
          << "Test failure on JSON Parse: pattern length for the configuration "
          << i << " was " << cl.configs[i]->pattern.size() << ", excpected "
          << pattern_lengths[i] << std::endl;
      return EXIT_FAILURE;
    }

    if (cl.configs[i]->count != counts[i]) {
      std::cerr << "Test failure on JSON Parse: counts for configuration " << i
                << " was "
                << ", expected " << counts[i] << std::endl;
      return EXIT_FAILURE;
    }

    if (!kernel.compare("gather") && cl.configs[i]->kernel.compare("gather")) {
      std::cerr << "Test failure on JSON Parse: user request kernel gather but "
                   "instead got other kernel "
                << cl.configs[i]->kernel << "." << std::endl;
      return EXIT_FAILURE;
    }

    if (!kernel.compare("scatter") &&
        cl.configs[i]->kernel.compare("scatter")) {
      std::cerr << "Test failure on JSON Parse: user requested kernel scatter "
                   "but instead got other kernel "
                << cl.configs[i]->kernel << "." << std::endl;
      return EXIT_FAILURE;
    }

    if (!kernel.compare("sg") && cl.configs[i]->kernel.compare("sg")) {
      std::cerr << "Test failure on JSON Parse: user requested kernel SG but "
                   "instead got other kernel "
                << cl.configs[i]->kernel << "." << std::endl;
      return EXIT_FAILURE;
    }

    for (size_t j = 0; j < pattern_lengths[i]; ++j) {
      if (patterns[i][j] != cl.configs[i]->pattern[j]) {
        std::cerr << "Test failure on JSON Parse: pattern mismatch at index "
                  << i << ", got value " << cl.configs[i]->pattern[j]
                  << " but expected " << patterns[i][j] << "." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

#ifndef JSON_SRC
  printf("JSON SRC Directory not defined!\n");
  return EXIT_FAILURE;
#else
  int argc_ = 2;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-f%s", JSON_SRC);

  std::cout << argv_[0] << " " << argv_[1] << std::endl;

  std::vector<size_t> pattern_lengths = {16, 16};
  std::vector<size_t> pattern1 = {1333, 0, 1, 2, 36, 37, 38, 72, 73, 74, 1296,
      1297, 1298, 1332, 1334, 1368};
  std::vector<size_t> pattern2 = {1333, 0, 1, 36, 37, 72, 73, 1296, 1297, 1332,
      1368, 1369, 2592, 2593, 2628, 2629};
  std::vector<size_t> counts = {1454647, 1454647};

  std::vector<std::vector<size_t>> patterns = {pattern1, pattern2};

  std::string kernel = "gather";

  if (json_test(2, kernel, pattern_lengths, patterns, counts, argc_, argv_) !=
      EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  free(argv_[0]);
  free(argv_[1]);

  return EXIT_SUCCESS;
#endif
}
