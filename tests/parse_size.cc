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
  asprintf(&argv_[1], "-p0,1,2,3,4,5,6,7");
  asprintf(&argv_[2], "-j4");
  asprintf(&argv_[3], "-kgather");

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

  std::vector<size_t> gold = {0, 1, 2, 3};

  if (cl.configs[0]->pattern.size() != 4) {
    std::cerr << "Test failure Size Limited Pattern: input pattern was not "
                 "truncated to the correct length!"
              << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < 4; i++) {
    if (gold[i] != cl.configs[0]->pattern[i]) {
      std::cerr << "Test failure Size Limited Pattern: input pattern does not "
                   "much parse pattern!"
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
