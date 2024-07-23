#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int random_test(long int seed, int argc, char **argv) {
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
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

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-p1,2,3,4");
  asprintf(&argv_[2], "-s");

  if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  asprintf(&argv_[2], "-s123");
  if (random_test(123, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  asprintf(&argv_[2], "-s456");
  if (random_test(456, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  asprintf(&argv_[2], "--random");
  if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  asprintf(&argv_[2], "--random=789");
  if (random_test(789, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (int i = 0; i < argc_; ++i)
    free(argv_[i]);
  free(argv_);

  return EXIT_SUCCESS;
}
