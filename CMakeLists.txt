#3.8 is specified since this is the first version with CUDA support
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

#Set default values
SET( USE_CUDA   0 CACHE BOOL "If set, builds with CUDA support")
SET( USE_OPENCL 0 CACHE BOOL "If set, builds with OpenCL support")
SET( USE_OPENMP 0 CACHE BOOL "If set, builds with OpenMP support")

#Check for support for at least one backend
IF( NOT( USE_CUDA OR USE_OPENCL OR USE_OPENMP))
    MESSAGE( FATAL_ERROR "You must build with support for at least one backend. Pass at least one of -DUSE_CUDA=1 -DUSE_OPENCL=1 or -DUSE_OPENMP=1 to cmake." )
ENDIF()

#Set cuda compiler
IF( USE_CUDA )
    SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    SET(CMAKE_CUDA_COMPILER_ENV_VAR "CUDAC")
ENDIF()

#Declare project languages
IF( USE_CUDA )
    PROJECT(sgbench C CXX CUDA)
ELSE()
    PROJECT(sgbench C CXX)
ENDIF()

#Debug function to check all the variables in the CMakeFile 
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# Use the 2011 standards for C and C++
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Turn warnings into errors
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

# Enable OpenCL 
IF ( USE_OPENCL )
	ADD_DEFINITIONS( -DUSE_OPENCL )     
	#Point the compiler to the include and library directories
	include_directories($ENV{OCL_INCL} opencl)
	#include_directories(/usr/lib/gcc/x86_64-linux-gnu/5/include/)
	link_directories($ENV{OCL_LIB})
	#Pull the OpenCL-specific files into the build
	file (GLOB BACKEND_C_FILES opencl/*.c)
	file (GLOB BACKEND_H_FILES opencl/*.h)

	#Copy over the OpenCL kernels used with the binary
	file (GLOB OCL_KERNELS opencl/*.cl)
	file(COPY ${OCL_KERNELS} include/sgtype.h
	     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kernels/)

	MESSAGE( "Using OpenCL" ) 
ENDIF()

# Enable OpenMP 
IF ( USE_OPENMP )
	#enable pragma regions in the code for OpenMP
	ADD_DEFINITIONS( -DUSE_OPENMP )
	#Standard includes and libraries are probably ok for standard OMP     
	include_directories(openmp)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

	#Pull the OpenMP-specific files and kernels into the build
	file (GLOB BACKEND_C_FILES openmp/*.c)
	file (GLOB BACKEND_H_FILES openmp/*.h)
    MESSAGE( "Using OpenMP" )
ENDIF()

# Enable CUDA 
IF( USE_CUDA )
    ADD_DEFINITIONS( -DUSE_CUDA )
    include_directories(cuda)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    SET(CMAKE_CUDA_FLAGS "${CUDA_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include" )   
 
    file (GLOB CUDA_CU_FILES cuda/*.cu)
    file (GLOB CUDA_C_FILES cuda/*.c)
    file (GLOB CUDA_H_FILES  cuda/*.h)
    message( "Using Cuda" )
ENDIF()

#Include the location of stddef.h
#include_directories(/usr/include/linux/)

#Glob for the source files and headers in the current directory and kernels
file (GLOB C_FILES src/*.c)
file (GLOB H_FILES include/*.h)
SET (SOURCE_FILES ${C_FILES} ${H_FILES} ${BACKEND_C_FILES} ${BACKEND_H_FILES} ${CUDA_CU_FILES} ${CUDA_C_FILES})

#Specify the executable and source files
add_executable(sgbench ${SOURCE_FILES})

#Print out debug info
#print_all_variables()

#Link with the appropriate libraries
IF ( USE_OPENCL )
	target_link_libraries (sgbench LINK_PUBLIC OpenCL)
ENDIF()
If ( USE_CUDA )
    find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda-9.2/lib64/)
    target_link_libraries (sgbench LINK_PUBLIC ${CUDART_LIBRARY})
ENDIF()

#Copy over the test scripts 
file (GLOB TEST_SCRIPTS tests/*.sh)
file(COPY ${TEST_SCRIPTS}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

