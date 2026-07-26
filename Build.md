# Building Spatter

Spatter uses CMake (currently ≥ 3.25) and requires a C++17-compatible compiler.

## Quick Start

```bash
cmake -B build [options]
cmake --build build -j
```

The compiled `spatter` binary will be placed in the `build/` directory in this example.

## Installation

```bash
cmake --install build --prefix /path/to/install
```

This installs the `spatter` and `gz_read` binaries to `<prefix>/bin/`.

## Spack Installation
You can also use Spack to install the default (OpenMP) and CUDA versions of Spatter. Please see the [Spack packages](https://packages.spack.io/package.html?name=spatter) for more information.

---

## CMake Options

| Option | Type | Default | Description |
|---|---|---|---|
| `USE_CUDA` | `BOOL` | `OFF` | Enable NVIDIA CUDA backend |
| `USE_HIP` | `BOOL` | `OFF` | Enable AMD HIP/ROCm backend |
| `USE_ONEAPI` | `BOOL` | `OFF` | Enable Intel OneAPI SYCL backend |
| `USE_OPENMP` | `BOOL` | `OFF` | Enable OpenMP threading |
| `USE_MPI` | `BOOL` | `OFF` | Enable MPI support |
| `CMAKE_BUILD_TYPE` | `STRING` | `Release` | Build type: `Release`, `Debug`, `RelWithDebInfo` |
| `CMAKE_CXX_COMPILER` | `STRING` | system default | C++ compiler to use |
| `SPATTER_ENABLE_NATIVE_ARCH` | `BOOL` | `ON` | Tune GNU/Clang builds for the host CPU (adds `-march=native`, or `-mcpu=native` where unsupported) so the gather/scatter kernels use the host ISA (e.g. AVX2/AVX-512). Disable for portable/reproducible or cross builds. |
| `SPATTER_ARCH_FLAGS` | `STRING` | `""` | Explicit architecture flags for GNU/Clang, e.g. `-march=sapphirerapids`. Takes precedence over `SPATTER_ENABLE_NATIVE_ARCH`. |

> **Note:** `USE_CUDA`, `USE_HIP`, and `USE_ONEAPI` are mutually exclusive. Only one GPU backend may be enabled at a time.

## JSON Support
JSON support via `nlohmann/json v3.11.2` is automatically fetched via CMake's `FetchContent`.

---

## Backend Build Examples

> **Note:** We recommend using the syntax `build_<backend>` to build Spatter backends, which makes it easier to distinguish different variants of the executable.


### Serial (CPU only)

Any supported C++ compiler works, and no backend flags are required.

```bash
cmake -B build
cmake --build build -j
```

### OpenMP

```bash
cmake -B build_omp -DUSE_OPENMP=ON
cmake --build build_omp -j
```

The OpenMP backend can be combined with the MPI backend:

```bash
cmake -B build_omp_mpi -DUSE_OPENMP=ON -DUSE_MPI=ON
cmake --build build_omp_mpi -j
```

### CUDA (NVIDIA GPUs)

The CUDA backend requires the CUDA Toolkit and a CUDA-capable compiler (`nvcc`) to be on `PATH`. CMake will auto-detect the installed CUDA toolkit.

```bash
cmake -B build_cuda -DUSE_CUDA=ON
cmake --build build_cuda -j
```
 If multiple CUDA versions are installed, you can point CMake at the preferred one as follows:

```bash
cmake -B build_cuda12 -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build_cuda12 -j
```

### HIP (AMD GPUs)

The HIP backend requires the ROCm Toolkit and a HIP-capable C++ compiler (`hipcc`) to be on `PATH`.  The default ROCm path is `/opt/rocm`, but this can be overriden with `-DHIP_PATH=<path>`.

```bash
cmake -B build_hip -DUSE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc
cmake --build build_hip -j
```

With a non-default ROCm installation:

```bash
cmake -B build_hip -DUSE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DHIP_PATH=/opt/<rocm-custom>
cmake --build build_hip -j
```

### OneAPI (Intel GPUs / SYCL)

The OneAPI backend requires the OneAPI Toolkit and a SYCL-capable C++ compiler (`icpx`) to be on `PATH`. You can typically set `icpx` to be on your path using environment variables or the `setvars.sh` script included with OneAPI installations.

```bash
cmake -B build_oneapi -DUSE_ONEAPI=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build_oneapi -j
```

---

## Supported Compilers

| Compiler | ID | Serial | OpenMP | CUDA | HIP | OneAPI |
|---|---|:---:|:---:|:---:|:---:|:---:|
| GCC | `GNU` | ✓ | ✓ | | | |
| Clang | `Clang` | ✓ | ✓ | | | |
| Intel (`icpx`) | `IntelLLVM` | ✓ | ✓ | | | ✓ |
| Cray CCE | `Cray` | ✓ | ✓ | | | |
| IBM XL | `XL` | ✓ | ✓ | | | |
| NVCC | `NVIDIA` | | | ✓ | | |
| `hipcc` | `ROCm (Clang)) | | | | ✓ | |

---

## Build Types

The default build type is `Release`. To change it:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```
or 
```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Debug builds include `-g -debug all`. GCC and Clang builds enable `-Wall -Wextra -Wconversion -pedantic-errors`.

---
