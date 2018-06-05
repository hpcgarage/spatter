cmake_minimum_required (VERSION 3.2)
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(sgbench C CXX)

# Use the 2011 standards for C and C++
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Turn warnings into errors
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
# Enable OpenMP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#Point the compiler to the include and library directories
include_directories($ENV{OCL_INCL})
link_directories($ENV{OCL_LIB})

add_executable(sgbench *.c kernels/*.c)

target_link_libraries (sgbench LINK_PUBLIC OpenCL)
