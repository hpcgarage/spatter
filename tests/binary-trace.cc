#include <iostream>

int gz_read_test() {
  char *command;

  int ret =
      asprintf(&command, "../gz_read -q -f %s/0.0.R.idx.gz", BINARY_TRACE_DIR);
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

#ifndef BINARY_TRACE_DIR
  std::cerr << "BINARY TRACE Directory not defined!" << std::endl;
  return EXIT_FAILURE;
#else
  if (gz_read_test() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif
}
