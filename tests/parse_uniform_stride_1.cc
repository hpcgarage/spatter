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

  strcpy(argv_[0], "./src/spatter-driver");
  strcpy(argv_[1], "-pUNIFORM:8:1");

  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc_, argv_, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1)
    std::cerr << "Test failure on Uniform Stride: Expected number of runs to "
                 "be 1, actually was "
              << cl.configs.size() << std::endl;

  if (cl.configs[0] == NULL)
    std::cerr << "Test failure on Uniform Stride: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;

  std::vector<size_t> validate = {0, 1, 2, 3, 4, 5, 6, 7};

  for (size_t i = 0; i < validate.size(); ++i)
    if (validate[i] != cl.configs[0]->pattern[i])
      return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
