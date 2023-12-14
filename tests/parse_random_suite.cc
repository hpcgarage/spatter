#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int random_test(int seed, int argc, char **argv) {
  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return
  }

  if (cl.configs.size() != 1) {
    std::cerr << "Test failure on Random Argument: Expected number of runs to "
                 "be 1, actually was "
              << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr
        << "Test failure on Random Argument: Failed to create or allocate "
           "a ConfigurationBase object"
        << std::endl;
    return EXIT_FAILURE;
  }

  // If this function is passed seed=-1, that means the seed it unset
  // and should be random. We'll only check the cases where it is set
  // explicitly.
  if (seed >= 0 && cl.configs[0]->seed != seed) {
    std::cerr << "Test failure on Random Argument: expected a random seed of "
              << seed << " but got a seed of " << cl.configs[0]->seed
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 3;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  for (size_t i = 0; i < argc_; ++i)
    argv_[i] = (char *)malloc(sizeof(char) * 1024);

  sprintf(argv_[0], "./src/spatter-driver");
  sprintf(argv_[1], "-p1,2,3,4");
  sprintf(argv_[2], "-l");

  if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  sprintf(argv_[2], "-l123");
  if (random_test(123, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  sprintf(argv_[2], "-l456");
  if (random_test(456, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  sprintf(argv_[2], "--random");
  if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  sprintf(argv_[2], "--random=789");
  if (random_test(789, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (size_t i = 0; i < argc_; ++i)
    free(argv_[i]);
  free(argv);

  return EXIT_SUCCESS;
}
