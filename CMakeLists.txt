# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(DisableIntree)

project(Spatter VERSION 2.2.0 LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" OFF)
option(USE_HIP "Enable HIP support" OFF)
option(USE_ONEAPI "Enable HIP support" OFF)

set(SELECTED_COUNT 0)

if(USE_CUDA)
    math(EXPR SELECTED_COUNT "${SELECTED_COUNT} + 1")
endif()

if(USE_ONEAPI)
    math(EXPR SELECTED_COUNT "${SELECTED_COUNT} + 1")
endif()

if(USE_HIP)
    math(EXPR SELECTED_COUNT "${SELECTED_COUNT} + 1")
endif()

# If more than one option is selected, throw an error
if(SELECTED_COUNT GREATER 1)
    message(FATAL_ERROR "Conflicting options detected: Only one of USE_CUDA, USE_ONEAPI, or USE_HIP can be enabled at a time.")
endif()

if(USE_HIP)
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "HIP C++ Compiler" FORCE)
    endif()
    enable_language(HIP)
    include(pkgs/HIPSupport)
else()
    if(USE_CUDA)
        include(pkgs/CUDASupport)
    endif()
endif()

include(GNUInstallDirs)
include(BuildType)
include(CompilerType)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(pkgs/JSONSupport)
include(pkgs/MPISupport)
include(pkgs/OpenMPSupport)
include(pkgs/CUDASupport)
include(pkgs/OneAPISupport)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -debug all")

if (APPLE)
        set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else ()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif ()
>>>>>>> spatter-devel

# Create gz_read executable
add_executable(gz_read standard-suite/binary-traces/gz_read.cc)
target_link_libraries(gz_read z)

install(TARGETS gz_read
        RUNTIME DESTINATION bin)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
