#include <iostream>
#include <vector>
#include <sycl/sycl.hpp> // OneAPI SYCL header

bool is_gpu_available() {
  try {
    sycl::device dev = sycl::device(sycl::default_selector{});
    return dev.is_gpu();
  } catch (...) {
    return false;
  }
}

int gpu_stream_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/basic-tests/gpu-stream-oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/basic-tests/cpu-stream.json");
  }

  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int gpu_ustride_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/basic-tests/gpu-ustride-oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/basic-tests/cpu-ustride.json");
  }

  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int gpu_amg_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/amg_gpu_oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/amg.json");
  }

  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}


int gpu_lulesh_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/lulesh_gpu_oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/lulesh.json");
  }

  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int gpu_nekbone_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/nekbone_gpu_oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/nekbone.json");
  }

  if (ret == -1 || system(command) != EXIT_SUCCESS) {
    std::cerr << "Test failure on " << command << std::endl;
    return EXIT_FAILURE;
  }

  free(command);
  return EXIT_SUCCESS;
}

int gpu_pennant_test(bool use_gpu) {
  char *command;

  int ret;
  if (use_gpu) {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/pennant_gpu_oneapi.json");
  } else {
    ret = asprintf(&command,
        "../spatter -b oneapi  -f "
        "../../standard-suite/app-traces/pennant.json");
  }

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

  bool use_gpu = is_gpu_available();
  std::cout << "GPU Available: " << (use_gpu ? "Yes" : "No (running CPU version)") << std::endl;

  if (gpu_stream_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_ustride_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_amg_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_lulesh_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_nekbone_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (gpu_pennant_test(use_gpu) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
