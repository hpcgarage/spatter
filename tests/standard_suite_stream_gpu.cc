#include <iostream>
#include <vector>

int gpu_stream_test() {
  char *command;

  int ret = asprintf(&command,
      "../spatter -b cuda  -f "
      "../../standard-suite/basic-tests/gpu-stream.json");
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

  if (gpu_stream_test() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
