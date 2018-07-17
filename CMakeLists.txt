cmake_minimum_required (VERSION 3.5)
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(sgbench C CXX)

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
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Turn warnings into errors
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")


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
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

	#Pull the OpenMP-specific files and kernels into the build
	file (GLOB BACKEND_C_FILES openmp/*.c)
	file (GLOB BACKEND_H_FILES openmp/*.h)
ENDIF()

#Include the location of stddef.h
#include_directories(/usr/include/linux/)

#Glob for the source files and headers in the current directory and kernels
file (GLOB C_FILES src/*.c)
file (GLOB H_FILES include/*.h)
set (SOURCE_FILES ${C_FILES} ${H_FILES} ${BACKEND_C_FILES} ${BACKEND_H_FILES})

#Specify the executable and source files
add_executable(sgbench ${SOURCE_FILES})

#Print out debug info
#print_all_variables()

#Link with the appropriate libraries
IF ( USE_OPENCL )
	target_link_libraries (sgbench LINK_PUBLIC OpenCL)
ENDIF()

#Copy over the test scripts 
file (GLOB TEST_SCRIPTS tests/*.sh)
file(COPY ${TEST_SCRIPTS}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

