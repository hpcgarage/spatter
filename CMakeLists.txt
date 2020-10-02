# 3.10 is specified since this is the first version with FindCUDAToolkit support
CMAKE_MINIMUM_REQUIRED (VERSION 3.10 FATAL_ERROR)

SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set target project name
SET (TRGT spatter)

# Set default values
SET (USE_CUDA 0 CACHE BOOL "If set, builds with CUDA support")
SET (USE_OPENCL 0 CACHE BOOL "If set, builds with OpenCL support")
SET (USE_OPENMP 0 CACHE BOOL "If set, builds with OpenMP support")
SET (USE_SERIAL 0 CACHE BOOL "If set, builds with Serial support")
SET (USE_PAPI 0 CACHE BOOL "If set, builds with PAPI support")
SET (USE_SYCL 0 CACHE BOOL "If set, builds with SYCL support")
SET (USE_CPU_HOST 0 CACHE BOOL "If set (with USE_SYCL), builds targeting host device")
SET (USE_FPGA_EMULATOR 0 CACHE BOOL "If set (with USE_SYCL), builds targeting FPGA emulation")
SET (USE_FPGA_REPORT 0 CACHE BOOL "If set (with USE_SYCL), builds targeting FPGA emulation with static report generation")
SET (USE_FPGA_PROFILING 0 CACHE BOOL "If set (with USE_SYCL), builds FPGA bitstream with vTune profiling enabled")
SET (USE_DEBUG_FLAG 0 CACHE BOOL "If set, compiles Spatter with -g flag for gdb debugging")

ADD_DEFINITIONS (-DSG_DATA_TYPE=double)

# Check for support for at least one backend
IF (NOT (USE_SYCL OR USE_CUDA OR USE_OPENCL OR USE_OPENMP OR USE_SERIAL))
    MESSAGE (
        FATAL_ERROR
            "You must build with support for at least one backend. Pass at least one of -D USE_SYCL=1 -DUSE_CUDA=1 -DUSE_OPENCL=1 -DUSE_SERIAL=1 or -DUSE_OPENMP=1 to cmake."
    )
ENDIF ()

# Set CUDA compiler
IF (USE_CUDA)
    SET (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    SET (CMAKE_CUDA_COMPILER_ENV_VAR "CUDAC")
ENDIF ()

IF (USE_DEBUG_FLAG)
	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
ENDIF ()

# Set SYCL compiler flags and definitions
# IF (USE_SYCL)
#    ADD_DEFINITIONS (-DUSE_SYCL)
#    IF (USE_FPGA_EMULATOR)
#	    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")
#	    ADD_DEFINITIONS (-DFPGA_EMULATOR)
#    ELSEIF (USE_FPGA_REPORT)
#	    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -fsycl-link -Xshardware")
#	    ADD_DEFINITIONS (-DFPGA_EMULATOR)
#    ELSEIF (USE_FPGA_BITSTREAM)
#	    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")
#    ELSEIF (USE_FPGA_PROFILING)
#	    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -Xshardware -Xsprofile")
#    ELSE () #Use CPU emulation by default
#	    message("Building for CPU host emulation")
#	    ADD_DEFINITIONS (-DCPU_HOST)
#    ENDIF ()
# ENDIF ()

# Declare project languages
IF (USE_CUDA)
    PROJECT (${TRGT} C CXX CUDA)
ELSEIF (USE_SYCL)
    PROJECT (${TRGT} C CXX)
ELSE ()
    PROJECT (${TRGT} C CXX)
ENDIF ()

# Debug function to check all the variables in the CMakeFile
MACRO (print_all_variables)
    MESSAGE (
        STATUS "print_all_variables------------------------------------------{"
    )
    GET_CMAKE_PROPERTY (_variableNames VARIABLES)
    FOREACH (_variableName ${_variableNames})
        MESSAGE (STATUS "${_variableName}=${${_variableName}}")
    ENDFOREACH ()
    MESSAGE (
        STATUS "print_all_variables------------------------------------------}"
    )
ENDMACRO ()


# Check that appropriate compiler versions are used
IF (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        MESSAGE (FATAL_ERROR "GCC version must be at least 4.9.1 to run the OpenMP 4.0 backend")
ENDIF ()


#Add specific flags for the CodeXL compiler
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
	SET(IBMXL_COMPILE_FLAGS "-qenablevmx -qtune=pwr9")
	SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${IBMXL_COMPILE_FLAGS}")
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=IBM")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Cray") 
    SET (OPTIMIZATIONS "-O3 -h vector3 -h cache3 -h scalar3")
    #SET (OPTIMIZATIONS "-G0 -O0")
    #SET (ALLWARNINGS "-Wall -Wextra -ansi -pedantic")
    #SET (ALLWARNINGS "-h msglevel_2")
    SET (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OPTIMIZATIONS} ${ALLWARNINGS} -hlist=m -hlist=d ")
    SET (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPTIMIZATIONS} -hlist=m -D__CRAYC__")
    SET (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Cray -D__CRAYC__")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel") 
    SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -xHost")
    SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -xHost")
ENDIF ()

 
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") 
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=GNU -O3 -g")
   #Use sanitize=address for debugging of access issues 
   #SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fsanitize=address")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang") 
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Clang")
ENDIF ()


#Add configure-specified flags
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")


#Check if Spatter is compiled on MacOSX. If so, use POSIX_MEMALIGN to allocate memory
IF (APPLE)
	ADD_DEFINITIONS (-DUSE_POSIX_MEMALIGN)
ENDIF ()

# Enable OpenCL
IF (USE_OPENCL)
    ADD_DEFINITIONS (-DUSE_OPENCL)
    # Point the compiler to the include and library directories
    INCLUDE_DIRECTORIES ($ENV{OCL_INCL} src/opencl)
    # include_directories(/usr/lib/gcc/x86_64-linux-gnu/5/include/)
    LINK_DIRECTORIES ($ENV{OCL_LIB})
    # Pull the OpenCL-specific files into the build
    FILE (GLOB BACKEND_C_FILES_CL src/opencl/*.c)
    FILE (GLOB BACKEND_H_FILES_CL src/opencl/*.h)

    # Copy over the OpenCL kernels used with the binary
    FILE (GLOB OCL_KERNELS src/opencl/*.cl)
    FILE (
        COPY
        ${OCL_KERNELS}
        include/sgtype.h
        DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/
    )

ENDIF ()

# Enable Serial backend
IF (USE_SERIAL)
    ADD_DEFINITIONS (-DUSE_SERIAL)
    INCLUDE_DIRECTORIES (src/serial)
    # Pull the serial files and kernels into the build
    FILE (GLOB BACKEND_C_FILES_SERIAL src/serial/*.c)
    FILE (GLOB BACKEND_H_FILES_SERIAL src/serial/*.h)
    MESSAGE ("Using serial backend")
ENDIF ()

# Enable OpenMP
IF (USE_OPENMP)
    FIND_PACKAGE (OpenMP REQUIRED)
    if(OpenMP_FOUND)
	# enable pragma regions in the code for OpenMP
        ADD_DEFINITIONS (-DUSE_OPENMP)
        ADD_DEFINITIONS (-DUSE_OMP_SIMD)
        INCLUDE_DIRECTORIES (src/openmp)

        # Pull the OpenMP-specific files and kernels into the build
        FILE (GLOB BACKEND_C_FILES_OMP src/openmp/*.c)
        FILE (GLOB BACKEND_H_FILES_OMP src/openmp/*.h)
    ENDIF ()
ENDIF ()

# Enable CUDA
IF (USE_CUDA)
    ADD_DEFINITIONS (-DUSE_CUDA)
    INCLUDE_DIRECTORIES (src/cuda)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    SET (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    IF (USE_OPENMP)
        SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF ()

    FILE (GLOB CUDA_CU_FILES src/cuda/*.cu)
    FILE (GLOB CUDA_C_FILES src/cuda/*.c)
    FILE (GLOB CUDA_H_FILES src/cuda/*.h)
ENDIF ()

IF (USE_SYCL)
    INCLUDE_DIRECTORIES (src/sycl)
    FILE (GLOB SYCL_CPP_FILES src/sycl/*.cpp)
    FILE (GLOB SYCL_HPP_FILES src/sycl/*.hpp src/sycl/*.h)
    # Set SYCL compiler flags and definitions
    ADD_DEFINITIONS (-DUSE_SYCL)
    IF (USE_FPGA_EMULATOR)
        #SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -fintelfpga)
        ADD_DEFINITIONS (-DFPGA_EMULATOR)
    ELSEIF (USE_FPGA_REPORT)
        #SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -fsycl-link -Xshardware")
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -fintelfpga)
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -fsycl-link)
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -Xshardware)
        ADD_DEFINITIONS (-DFPGA_EMULATOR)
    ELSEIF (USE_FPGA_BITSTREAM)
        #SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -Xshardware")
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -fintelfpga)
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -Xshardware)
    ELSEIF (USE_FPGA_PROFILING)
        #SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -Xshardware -Xsprofile")
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -fintelfpga)
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -Xshardware)
        SET_SOURCE_FILES_PROPERTIES(SYCL_CPP_FILES PROPERTIES COMPILE_FLAGS -Xsprofile)
    ELSE () #Use CPU emulation by default
        message("Building for CPU host emulation")
        ADD_DEFINITIONS (-DCPU_HOST)
    ENDIF ()
ENDIF ()

IF (USE_PAPI) 
    SET (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    INCLUDE_DIRECTORIES (src/papi)
    ADD_DEFINITIONS (-DUSE_PAPI)
    #Use the included FindPAPI cmake module
    INCLUDE (FindPAPI)
    FIND_PACKAGE (PAPI REQUIRED)
    #message("Found PAPI libraries at ${PAPI_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(${PAPI_INCLUDE_DIRS})
    FILE (GLOB PAPI_C_FILES src/papi/*.c)
    FILE (GLOB PAPI_H_FILES src/papi/*.h)
ENDIF ()

# Include the location of stddef.h include_directories(/usr/include/linux/)

# Glob for the source files and headers in the current directory and kernels
FILE (GLOB C_FILES src/*.c)
FILE (GLOB H_FILES include/*.h)
SET (
    SOURCE_FILES
    ${C_FILES}
    ${H_FILES}
    ${BACKEND_C_FILES_CL}
    ${BACKEND_H_FILES_CL}
    ${BACKEND_C_FILES_OMP}
    ${BACKEND_H_FILES_OMP}
    ${BACKEND_C_FILES_SERIAL}
    ${BACKEND_H_FILES_SERIAL}
    ${CUDA_CU_FILES}
    ${CUDA_C_FILES}
    ${PAPI_C_FILES}
    ${PAPI_H_FILES}
    ${SYCL_CPP_FILES}
    #${SYCL_HPP_FILES}
)

# Specify the executable and source files
ADD_EXECUTABLE (${TRGT} ${SOURCE_FILES})

#Prefer C11 standard to allow for loop initialization and aligned_alloc
set_property(TARGET ${TRGT} PROPERTY C_STANDARD 11)

# Print out debug info print_all_variables()

# Link with the appropriate libraries
IF (USE_OPENMP)
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenMP::OpenMP_CXX)
ENDIF()

IF (USE_OPENCL)
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenCL)
ENDIF ()

# Link sycl if using
IF (USE_SYCL AND NOT (USE_FPGA_BITSTREAM OR USE_FPGA_PROFILING))
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC sycl)
ENDIF ()

# Link argtable

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/argtable3/src")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/argtable3")
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/external/argtable3")
TARGET_LINK_LIBRARIES(${TRGT} LINK_PUBLIC argtable3_static)


# Link math library for json
TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC m)

IF (USE_CUDA)
    FIND_LIBRARY (
        CUDART_LIBRARY cudart
        HINTS
            /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/
    )
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC ${CUDART_LIBRARY})
ENDIF ()

#Include PAPI libraries, if defined
IF (USE_PAPI)
    TARGET_LINK_LIBRARIES (spatter LINK_PUBLIC ${PAPI_LIBRARIES})
    # Explicitly link libdl because the Cray compiler refuses to link papi dynaimically...
    # (Static libraries carry no dependency info, and libpapi depends on libdl)
    TARGET_LINK_LIBRARIES (spatter LINK_PUBLIC dl)
ENDIF ()

# Copy over the test scripts
FILE (GLOB TEST_SCRIPTS tests/*.sh)
FILE (COPY ${TEST_SCRIPTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Add a command to put some important info in environment variables 
# to be output by spatter
message (STATUS "compiler version is ${CMAKE_C_COMPILER_VERSION}")
message (STATUS "compiler is ${CMAKE_C_COMPILER}")
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C=\"${CMAKE_C_COMPILER}\"")
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C_VER=\"${CMAKE_C_COMPILER_VERSION}\"")


# Build tests
mark_as_advanced( BUILD_TESTS )
set( BUILD_TESTS true CACHE BOOL "Tests build target available if true" )
if( BUILD_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif( BUILD_TESTS )


# Note - This file can be reformatted with cmake-format CMakeLists.txt -i
# --command-case=upper --keyword-case=upper --dangle-parens --tab-size=4
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do not
# have cmake-format installed, install it with pip install --user cmake_format
