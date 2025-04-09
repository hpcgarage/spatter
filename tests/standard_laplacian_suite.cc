#include <iostream>

int laplacian_test() {
  int dimension = 1, pseudo_order = 1, problem_size = 100;

  for (int i = 0; i < 3; i++) {
    char *command;

    int ret = asprintf(&command, "../spatter -pLAPLACIAN:%d:%d:%d",
        dimension, pseudo_order, problem_size);
    if (ret == -1 || system(command) != EXIT_SUCCESS) {
      std::cerr << "Test failure on " << command << std::endl;
      return EXIT_FAILURE;
    }

    dimension += 1;
    pseudo_order += 1;
    problem_size += 10;
    free(command);
  }

  return EXIT_SUCCESS;
}

int laplacian_test_delta() {
  for (int delta = 1; delta < 100; delta *= 2) {
    char *command;

    int ret = asprintf(
        &command, "../spatter -pLAPLACIAN:2:2:100 -d%d", delta);
    if (ret == -1 || system(command) != EXIT_SUCCESS) {
      std::cerr << "Test failure on " << command << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  if (laplacian_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (laplacian_test_delta() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
