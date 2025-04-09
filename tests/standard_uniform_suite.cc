#include <iostream>
#include <vector>

int uniform_test_length_gap() {
  int length = 8, gap = 2;
  for (int i = 0; i < 8; i++) {
    char *command;

    int ret = asprintf(
        &command, "../spatter -pUNIFORM:%d:%d", length, gap);
    if (ret == -1 || system(command) != EXIT_SUCCESS) {
      std::cerr << "Test failure on " << command << std::endl;
      return EXIT_FAILURE;
    }

    length *= 2;
    gap += 2;
    free(command);
  }

  return EXIT_SUCCESS;
}

int uniform_test_delta() {
  for (int delta = 1; delta < 100; delta *= 2) {
    char *command;

    int ret =
        asprintf(&command, "../spatter -pUNIFORM:8:4 -d%d", delta);
    if (ret == -1 || system(command) != EXIT_SUCCESS) {
      std::cerr << "Test failure on " << command << std::endl;
      return EXIT_FAILURE;
    }

    free(command);
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  if (uniform_test_length_gap() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (uniform_test_delta() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
