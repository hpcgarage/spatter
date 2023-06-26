# 3.10 is specified since this is the first version with FindCUDAToolkit support
CMAKE_MINIMUM_REQUIRED (VERSION 3.10 FATAL_ERROR)
project(spatter C CXX)

SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set target project name
SET (TRGT spatter)

# Use for debugging. TODO: Set this automatically
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Use ASAN when debugging
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

#Make backend and compiler variables case insensitive
string( TOLOWER "${BACKEND}" BACKEND )
string( TOLOWER "${COMPILER}" COMPILER )

#Set backend permitted values
set(spatterBackends serial openmp cuda)

#Check for backend variable set in user cmake call
if(NOT BACKEND IN_LIST spatterBackends)
    message(FATAL_ERROR "backend must be one of ${spatterBackends}")
endif()

#Set compiler permitted values
set(spatterCompilers cray gnu clang intel xl nvcc)

#Check for compiler variable set in user cmake call
if(NOT COMPILER IN_LIST spatterCompilers)
    message(FATAL_ERROR "compiler must be one of ${compilerBackends}")
endif()

ADD_DEFINITIONS (-DSG_DATA_TYPE=double)

# Check for support for at least one backend
IF ("${BACKEND}" STREQUAL "")
    MESSAGE (
        FATAL_ERROR
            "You must build with support for at least one backend. Pass at least one of -DBACKEND=serial, openmp, or cuda to cmake."
    )
ENDIF ()

# Check for support for at least one compiler
IF ("${COMPILER}" STREQUAL "")
    MESSAGE (
        FATAL_ERROR
            "You must build with support for at least one compiler. Pass at least one of -DCOMPILER=cray, gnu, intel, xl, or nvidia to cmake."
    )
ENDIF ()


# Set backend specific flags
IF ("${BACKEND}" STREQUAL "cuda")
    SET(CMAKE_CXX_COMPILER "${COMPILER}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -03")
    #TODO: How will user specify CUDA arch
    SET(CMAKE_CUDA_ARCHITECTURES "70")
ENDIF ()

# Set CUDA compiler
IF ("${BACKEND}" STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit)
    SET (CMAKE_CUDA_COMPILER_ENV_VAR "CUDAC")
ENDIF ()

# Declare project languages
IF ("${BACKEND}" STREQUAL "cuda")
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
    SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -xHost -qopenmp")
    SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -xHost -qopenmp")
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
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include -I${CMAKE_CURRENT_SOURCE_DIR}/external/argtable3")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include -I${CMAKE_CURRENT_SOURCE_DIR}/external/argtable3")


#Check if Spatter is compiled on MacOSX. If so, use POSIX_MEMALIGN to allocate memory
IF (APPLE)
	ADD_DEFINITIONS (-DUSE_POSIX_MEMALIGN)
ENDIF ()

# Enable OpenCL
IF ("${BACKEND}" STREQUAL "opencl")
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
    MESSAGE ("Using OpenCL backend")
ENDIF ()

# Enable Serial backend
IF ("${BACKEND}" STREQUAL "serial")
    ADD_DEFINITIONS (-DUSE_SERIAL)
    INCLUDE_DIRECTORIES (src/serial)
    # Pull the serial files and kernels into the build
    FILE (GLOB BACKEND_C_FILES_SERIAL src/serial/*.c)
    FILE (GLOB BACKEND_H_FILES_SERIAL src/serial/*.h)
    MESSAGE ("Using serial backend")
ENDIF ()

# Enable OpenMP
IF ("${BACKEND}" STREQUAL "openmp")
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
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    MESSAGE ("Using OpenMP backend")
ENDIF ()

# Enable CUDA
IF ("${BACKEND}" STREQUAL "cuda")
    ADD_DEFINITIONS (-DUSE_CUDA)
    INCLUDE_DIRECTORIES (src/cuda)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    SET (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )
    IF ("${BACKEND}" STREQUAL "openmp")
        SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF ()

    FILE (GLOB CUDA_CU_FILES src/cuda/*.cu)
    FILE (GLOB CUDA_C_FILES src/cuda/*.c)
    FILE (GLOB CUDA_H_FILES src/cuda/*.h)
    MESSAGE ("Using CUDA backend")
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

# Include amalgamated argtable files
FILE (GLOB EXTERNAL_C_FILES external/argtable3/*.c)
FILE (GLOB EXTERNAL_H_FILES external/argtable3/*.h)

# Glob for the source files and headers in the current directory and kernels
FILE (GLOB C_FILES src/*.c)
FILE (GLOB H_FILES src/include/*.h)
SET (
    SOURCE_FILES
    ${C_FILES}
    ${H_FILES}
    ${EXTERNAL_C_FILES}
    ${EXTERNAL_H_FILES}
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
)

# Specify the executable and source files
ADD_EXECUTABLE (${TRGT} ${SOURCE_FILES})

#Prefer C11 standard to allow for loop initialization and aligned_alloc
IF (NOT "${BACKEND}" STREQUAL "cuda")
	set_property(TARGET ${TRGT} PROPERTY C_STANDARD 11)
ENDIF()

# Print out debug info print_all_variables()

# Link with the appropriate libraries
IF ("${BACKEND}" STREQUAL "openmp")
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenMP::OpenMP_CXX)
ENDIF()

IF ("${BACKEND}" STREQUAL "opencl")
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenCL)
ENDIF ()

# Link math library for json
TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC m)

IF ("${BACKEND}" STREQUAL "cuda")
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC CUDA::cudart)
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
if ( "${BACKEND}" STREQUAL "openmp" OR "${BACKEND}" STREQUAL "serial" )
    mark_as_advanced( BUILD_TESTS )
    set( BUILD_TESTS true CACHE BOOL "Tests build target available if true" )
    if( BUILD_TESTS )
        enable_testing()
        add_subdirectory( tests )
    endif()
endif()

# Validation flag
SET (VALIDATE_DATA 0 CACHE BOOL "Performs additional validation")
IF ("${BACKEND}" STREQUAL "openmp" OR "${BACKEND}" STREQUAL "cuda")
    IF (VALIDATE_DATA)
        ADD_DEFINITIONS(-DVALIDATE)
    ENDIF ()
ENDIF ()


# Note - This file can be reformatted with cmake-format CMakeLists.txt -i
# --command-case=upper --keyword-case=upper --dangle-parens --tab-size=4
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do not
# have cmake-format installed, install it with pip install --user cmake_format