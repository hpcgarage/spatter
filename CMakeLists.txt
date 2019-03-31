# 3.8 is specified since this is the first version with CUDA support
CMAKE_MINIMUM_REQUIRED (VERSION 3.8 FATAL_ERROR)

#Set target project name
SET (TRGT spatter)

# Set default values
SET (USE_CUDA 0 CACHE BOOL "If set, builds with CUDA support")
SET (USE_OPENCL 0 CACHE BOOL "If set, builds with OpenCL support")
SET (USE_OPENMP 0 CACHE BOOL "If set, builds with OpenMP support")

ADD_DEFINITIONS (-DSG_DATA_TYPE=double)

# Check for support for at least one backend
IF (NOT (USE_CUDA OR USE_OPENCL OR USE_OPENMP))
    MESSAGE (
        FATAL_ERROR
            "You must build with support for at least one backend. Pass at least one of -DUSE_CUDA=1 -DUSE_OPENCL=1 or -DUSE_OPENMP=1 to cmake."
    )
ENDIF ()

# Set cuda compiler
IF (USE_CUDA)
    SET (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    SET (CMAKE_CUDA_COMPILER_ENV_VAR "CUDAC")
ENDIF ()

# Declare project languages
IF (USE_CUDA)
    PROJECT (${TRGT} C CXX CUDA)
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

SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

#Add specific flags for the CodeXL compiler
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
	SET(IBMXL_COMPILE_FLAGS "-qenablevmx -qtune=pwr9")
	SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${IBMXL_COMPILE_FLAGS}")
ENDIF ()


# Enable OpenCL
IF (USE_OPENCL)
    ADD_DEFINITIONS (-DUSE_OPENCL)
    # Point the compiler to the include and library directories
    INCLUDE_DIRECTORIES ($ENV{OCL_INCL} opencl)
    # include_directories(/usr/lib/gcc/x86_64-linux-gnu/5/include/)
    LINK_DIRECTORIES ($ENV{OCL_LIB})
    # Pull the OpenCL-specific files into the build
    FILE (GLOB BACKEND_C_FILES_CL opencl/*.c)
    FILE (GLOB BACKEND_H_FILES_CL opencl/*.h)

    # Copy over the OpenCL kernels used with the binary
    FILE (GLOB OCL_KERNELS opencl/*.cl)
    FILE (
        COPY
        ${OCL_KERNELS}
        include/sgtype.h
        DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/
    )

    MESSAGE ("Using OpenCL")
ENDIF ()

# Enable OpenMP
IF (USE_OPENMP)
    FIND_PACKAGE (OpenMP REQUIRED)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    # enable pragma regions in the code for OpenMP
    ADD_DEFINITIONS (-DUSE_OPENMP)
    ADD_DEFINITIONS (-DUSE_OMP_SIMD)
    INCLUDE_DIRECTORIES (openmp)

    # Pull the OpenMP-specific files and kernels into the build
    FILE (GLOB BACKEND_C_FILES_OMP openmp/*.c)
    FILE (GLOB BACKEND_H_FILES_OMP openmp/*.h)
    MESSAGE ("Using OpenMP")
ENDIF ()

# Enable CUDA
IF (USE_CUDA)
    ADD_DEFINITIONS (-DUSE_CUDA)
    INCLUDE_DIRECTORIES (cuda)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    SET (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    IF (USE_OPENMP)
        SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF ()

    FILE (GLOB CUDA_CU_FILES cuda/*.cu)
    FILE (GLOB CUDA_C_FILES cuda/*.c)
    FILE (GLOB CUDA_H_FILES cuda/*.h)
    MESSAGE ("Using Cuda")
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
    ${CUDA_CU_FILES}
    ${CUDA_C_FILES}
)

# Specify the executable and source files
ADD_EXECUTABLE (${TRGT} ${SOURCE_FILES})

# Print out debug info print_all_variables()

# Link with the appropriate libraries
IF (USE_OPENCL)
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenCL)
ENDIF ()

IF (USE_CUDA)
    FIND_LIBRARY (
        CUDART_LIBRARY cudart
        HINTS
            /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda-9.2/lib64/
    )
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC ${CUDART_LIBRARY})
ENDIF ()

# Copy over the test scripts
FILE (GLOB TEST_SCRIPTS tests/*.sh)
FILE (COPY ${TEST_SCRIPTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Note - This file can be reformatted with cmake-format CMakeLists.txt -i
# --command-case=upper --keyword-case=upper --dangle-parens --tab-size=4
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do not
# have cmake-format installed, install it with pip install --user cmake_format
