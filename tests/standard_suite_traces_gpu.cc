#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

const std::vector<std::string> SUPPORTED_BACKENDS = {"cuda", "hip"};

bool is_supported_backend(const std::string& backend) {
    return std::find(SUPPORTED_BACKENDS.begin(), SUPPORTED_BACKENDS.end(), backend) != SUPPORTED_BACKENDS.end();
}

int gpu_stream_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/basic-tests/gpu-stream.json", 
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int gpu_ustride_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/basic-tests/gpu-ustride.json",
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int gpu_amg_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/app-traces/amg_gpu.json",
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int gpu_lulesh_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/app-traces/lulesh_gpu.json",
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int gpu_nekbone_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/app-traces/nekbone_gpu.json",
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int gpu_pennant_test(const std::string& backend) {
  char *command;
  int ret = asprintf(&command,
      "../spatter -b %s -f "
      "../../standard-suite/app-traces/pennant_gpu.json",
      backend.c_str());
  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    free(command);
    return EXIT_FAILURE;
  }
  free(command);
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {

  std::string backend = "cuda"; // Default backend if not specified
  
  if (argc > 1) {
    backend = argv[1];
    if (!is_supported_backend(backend)) {
      std::cerr << "Error: Unsupported backend '" << backend << "'" << std::endl;
      std::cerr << "Supported backends: ";
      for (size_t i = 0; i < SUPPORTED_BACKENDS.size(); ++i) {
        std::cerr << SUPPORTED_BACKENDS[i];
        if (i < SUPPORTED_BACKENDS.size() - 1) std::cerr << ", ";
      }
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  std::cout << "Running GPU tests with backend: " << backend << std::endl;
  
  if (gpu_stream_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_ustride_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_amg_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_lulesh_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_nekbone_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_pennant_test(backend) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  
  std::cout << "All GPU tests completed successfully with " << backend << " backend!" << std::endl;
  return EXIT_SUCCESS;
}
