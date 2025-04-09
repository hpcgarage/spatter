#include <iostream>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int ms1_test(size_t pattern_length, int gap_location_count,
    std::vector<size_t> gap_locations, int gap_size_count,
    std::vector<int> gap_sizes, int argc, char **argv) {
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr << "Test failure on MS1: Expected number of runs to "
                 "be "
              << 1 << ", actually was " << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr << "Test failure on MS1: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (gap_location_count > 1 && gap_size_count != gap_location_count &&
      gap_size_count != 1) {
    std::cerr << "Test failure on MS1: specified number of gap locations is "
                 "inconsistent with number of gap sizes."
              << std::endl;
    return EXIT_FAILURE;
  }

  int current_value = -1;
  int gap_index = 0;
  for (size_t i = 0; i < pattern_length; ++i) {
    if (gap_index < gap_location_count && gap_locations[gap_index] == i) {
      if (gap_size_count > 1)
        current_value += gap_sizes[gap_index];
      else
        current_value += gap_sizes[0];
      gap_index++;
    } else
      current_value++;

    if ((unsigned long)current_value != cl.configs[0]->pattern[i]) {
      std::cerr << "Test failure on MS1: patterns do not match!\nGot pattern: ";

      std::cerr << "[";
      for (size_t j = 0; j < pattern_length; ++j) {
        std::cerr << cl.configs[0]->pattern[j];
        if (j != pattern_length - 1)
          std::cerr << " ";
      }
      std::cerr << "]";

      std::cerr << "Got value " << cl.configs[0]->pattern[i]
                << ", expected value " << current_value
                << "\nYou can find the expected patterns in comments in the "
                   "parse_m1_suite.cc file"
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

  for (size_t i = 8; i <= 64; i *= 2) {
    // Uniform gap sizes
    // [ 9, 10, 11, 12, 13, 14, 15, 16 ]
    optind = 1;
    asprintf(&argv_[1], "-pMS1:%ld:%s:%d", i, "0", 10);
    std::vector<size_t> gap_locations = {0};
    std::vector<int> gap_sizes = {10};
    if (ms1_test(i, 1, gap_locations, 1, gap_sizes, argc_, argv_) !=
        EXIT_SUCCESS)
      return EXIT_FAILURE;

    // [ 0, 5, 6, 7, 8, 13, 14, 19]
    optind = 1;
    asprintf(&argv_[1], "-pMS1:%ld:%s:%d", i, "1, 5, 7", 5);
    std::vector<size_t> gap_locations1 = {1, 5, 7};
    std::vector<int> gap_sizes1 = {5};
    if (ms1_test(i, 3, gap_locations1, 1, gap_sizes1, argc_, argv_) !=
        EXIT_SUCCESS)
      return EXIT_FAILURE;

    // Variable gap sizes
    // [ 4, 5, 14, 15, 16, 17, 18, 19 ]
    optind = 1;
    asprintf(&argv_[1], "-pMS1:%ld:%s:%s", i, "0, 2", "5, 9");
    std::vector<size_t> gap_locations2 = {0, 2};
    std::vector<int> gap_sizes2 = {5, 9};
    if (ms1_test(i, 2, gap_locations2, 2, gap_sizes2, argc_, argv_) !=
        EXIT_SUCCESS)
      return EXIT_FAILURE;

    // [ 0, 4, 5, 6, 7, 15, 16, 36 ]
    optind = 1;
    asprintf(&argv_[1], "-pMS1:%ld:%s:%s", i, "1, 5, 7", "3, 8, 20");
    std::vector<size_t> gap_locations3 = {1, 5, 7};
    std::vector<int> gap_sizes3 = {3, 8, 20};
    if (ms1_test(i, 3, gap_locations3, 3, gap_sizes3, argc_, argv_) !=
        EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
