#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 2;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  for (int i = 0; i < argc_; i++)
    argv_[i] = (char *)malloc(sizeof(char) * 1024);

  strncpy(argv_[0], "./spatter", 1024);
  strncpy(argv_[1], "-pUNIFORM:8:4:NR", 1024);

  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc_, argv_, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr << "Test failure on Uniform Stride: Expected number of runs to "
                 "be 1, actually was "
              << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr << "Test failure on Uniform Stride: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> expected_pattern = {0, 4, 8, 12, 16, 20, 24, 28};
  for (size_t i = 0; i < expected_pattern.size(); ++i)
    if (expected_pattern[i] != cl.configs[0]->pattern[i])
      return EXIT_FAILURE;

  size_t expected_delta = 8 * 4;
  if (cl.configs[0]->delta != expected_delta)
    return EXIT_FAILURE;

  free(argv_[0]);
  free(argv_[1]);
  free(argv_);

  return EXIT_SUCCESS;
}
