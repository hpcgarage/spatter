# Hermes/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(DisableIntree)

project(Spatter VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

include(BuildType)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include(pkgs/JSONSupport)
include(pkgs/MPISupport)
include(pkgs/CUDASupport)

add_subdirectory(src)

