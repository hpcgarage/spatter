#include <iostream>
#include <vector>

int tenstorrent_ustride_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -b tenstorrent  -f "
      "../../standard-suite/basic-tests/tenstorrent-ustride.json");
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

  if (tenstorrent_ustride_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
