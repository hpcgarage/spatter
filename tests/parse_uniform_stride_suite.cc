#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int uniform_stride_test(
    size_t pattern_length, size_t stride, int argc, char **argv) {
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
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

  for (size_t i = 0; i < pattern_length; ++i) {
    if (cl.configs[0]->pattern[i] != i * stride) {
      std::cerr
          << "Test failure on Uniform Stride with parameters:\nIndex Length: "
          << pattern_length << "\nStride: " << stride << std::endl;
      std::cerr << "Element at index " << i << " expected " << i * stride
                << ", actually was " << cl.configs[0]->pattern[i] << "."
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 3;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  for (unsigned int i = 1; i <= 16; i *= 2) {
    for (unsigned int j = 8; j <= 64; j *= 2) {
      optind = 1;
      asprintf(&argv_[1], "-pUNIFORM:%d:%d", j, i);
      if (uniform_stride_test(j, i, argc_ - 1, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    }

    for (int j = 8; j <= 64; j *= 2) {
      optind = 1;
      asprintf(&argv_[1], "--pattern=UNIFORM:%d:%d", j, i);
      if (uniform_stride_test(j, i, argc_ - 1, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    }
    for (int j = 8; j <= 64; j *= 2) {
      optind = 1;
      asprintf(&argv_[1], "-p");
      asprintf(&argv_[2], "UNIFORM:%d:%d", j, i);
      if (uniform_stride_test(j, i, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
