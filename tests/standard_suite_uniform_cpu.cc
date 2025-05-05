#include <iostream>
#include <vector>

int cpu_ustride_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -f "
      "../../standard-suite/basic-tests/cpu-ustride.json");
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

  if (cpu_ustride_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
