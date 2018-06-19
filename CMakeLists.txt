cmake_minimum_required (VERSION 3.5)
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

#Glob for the source files and headers in the current directory and kernels
file (GLOB C_FILES *.c kernels/*.c)
file (GLOB H_FILES *.h)
set (SOURCE_FILES ${C_FILES} ${H_FILES})

#Specify the executable and source files
add_executable(sgbench ${SOURCE_FILES})

#Link with OpenCL
target_link_libraries (sgbench LINK_PUBLIC OpenCL)

#Copy over the OpenCL kernels used with the binary
file (GLOB OCL_KERNELS kernels/*.cl)
file(COPY ${OCL_KERNELS}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kernels/)

#Copy over the test scripts 
file (GLOB TEST_SCRIPTS tests/*.sh)
file(COPY ${TEST_SCRIPTS}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
