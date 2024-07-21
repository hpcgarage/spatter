#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 4;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-uUNIFORM:8:1");
  asprintf(&argv_[2], "-gUNIFORM:8:1");
  asprintf(&argv_[3], "-kSG");

  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc_, argv_, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr
        << "Test failure on Concurrent Pattern: Expected number of runs to "
           "be 1, actually was "
        << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr
        << "Test failure on Concurrent Pattern: Failed to create or allocate "
           "a ConfigurationBase object"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> gold = {0, 1, 2, 3, 4, 5, 6, 7};

  for (int i = 0; i < 8; i++) {
    if (gold[i] != cl.configs[0]->pattern_scatter[i]) {
      std::cerr << "Test failure on Concurrent Pattern: input "
                   "pattern-scatter does not match parsed pattern-scatter!"
                << std::endl;
      return EXIT_FAILURE;
    }

    if (gold[i] != cl.configs[0]->pattern_gather[i]) {
      std::cerr << "Test failure on Concurrent Pattern: input pattern-gather "
                   "does not match parsed pattern-gather!"
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
