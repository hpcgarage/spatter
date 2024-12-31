#include <iostream>
#include <vector>

int cpu_amg_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -f ../../standard-suite/app-traces/amg.json");
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int cpu_lulesh_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -f ../../standard-suite/app-traces/lulesh.json");
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int cpu_nekbone_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -f ../../standard-suite/app-traces/nekbone.json");
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int cpu_pennant_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -f ../../standard-suite/app-traces/pennant.json");
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

  if (cpu_amg_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (cpu_lulesh_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (cpu_nekbone_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (cpu_pennant_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
