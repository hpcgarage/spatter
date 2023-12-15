# Hermes/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(DisableIntree)

project(Spatter VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

include(BuildType)
include(CompilerType)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include(pkgs/JSONSupport)
include(pkgs/MPISupport)
include(pkgs/OpenMPSupport)
include(pkgs/CUDASupport)

# Create gz_read executable
add_executable(gz_read standard-suite/binary-traces/gz_read.cc)
target_link_libraries(gz_read z)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
