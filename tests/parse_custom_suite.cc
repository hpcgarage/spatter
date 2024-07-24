#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int custom_test(
    size_t pattern_length, std::vector<size_t> values, int argc, char **argv) {
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr << "Test failure on Custom Pattern: Expected number of runs to "
                 "be 1, actually was "
              << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr << "Test failure on Custom Pattern: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < pattern_length; ++i) {
    if (values[i] != cl.configs[0]->pattern[i]) {
      std::cerr << "Test failure on Custom Pattern: input pattern does not "
                   "match parsed pattern!"
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 2;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-p1,2,3,4,5,6,7,8,9");

  std::vector<size_t> testValues = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (custom_test(8, testValues, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  asprintf(&argv_[1], "-p0");
  std::vector<size_t> testValues1 = {0};
  if (custom_test(1, testValues1, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  free(argv_[0]);
  free(argv_[1]);

  return EXIT_SUCCESS;
}
