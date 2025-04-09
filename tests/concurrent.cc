#include <iostream>

int uniform_concurrent_test() {
  for (int i = 1; i <= 10; i++) {
    char *command;

    int ret = asprintf(&command,
        "../spatter -kSG -uUNIFORM:%d:%d -gUNIFORM:%d:%d ", i, i, i,
        i);
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

  if (uniform_concurrent_test() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
