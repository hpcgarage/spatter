# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(DisableIntree)

project(Spatter VERSION 2.0.0 LANGUAGES CXX)

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

install(TARGETS gz_read
        RUNTIME DESTINATION bin)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
