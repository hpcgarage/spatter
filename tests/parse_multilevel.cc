#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int multilevel_test(int multigather) {
  int argc_ = 4;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-pUNIFORM:8:1");

  if (multigather) {
    asprintf(&argv_[2], "-gUNIFORM:8:1");
    asprintf(&argv_[3], "-kMultiGather");
  } else {
    asprintf(&argv_[2], "-uUNIFORM:8:1");
    asprintf(&argv_[3], "-kMultiScatter");
  }

  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc_, argv_, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr
        << "Test failure on Multilevel Pattern: Expected number of runs to "
           "be 1, actually was "
        << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr
        << "Test failure on Multilevel Pattern: Failed to create or allocate "
           "a ConfigurationBase object"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> gold = {0, 1, 2, 3, 4, 5, 6, 7};

  for (int i = 0; i < 8; i++) {
    if (gold[i] != cl.configs[0]->pattern[i]) {
      std::cerr << "Test failure on Multilevel Pattern: input pattern does "
                   "not match parsed pattern!"
                << std::endl;
      return EXIT_FAILURE;
    }

    if (multigather) {
      if (gold[i] != cl.configs[0]->pattern_gather[i]) {
        std::cerr << "Test failure on Multilevel Pattern: input "
                     "pattern-gather does not match parsed pattern-gather!"
                  << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      if (gold[i] != cl.configs[0]->pattern_scatter[i]) {
        std::cerr << "Test failure on Multilevel Pattern: input "
                     "pattern-scatter does not match parsed pattern-scatter!"
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  for (int i = 0; i < argc_; i++) {
    free(argv_[i]);
  }
  free(argv_);

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // MultiGather Test
  if (multilevel_test(1) < 0)
    return EXIT_FAILURE;

  // MultScatter Test
  if (multilevel_test(0) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
