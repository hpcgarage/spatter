# 3.18 is specified to include FindCUDAToolkit support updates
cmake_minimum_required (VERSION 3.18 FATAL_ERROR)

add_definitions (-DSG_DATA_TYPE=double)

#########################
# Configuration Options #
#########################

option(BACKEND "Which backend to use")
option(COMPILER "Which compiler to use. Sensible defaults are used if not specified.")
option(JIT "Whether to use JIT compilation for backends that support it (Only CUDA currently. HIP soon.)" ON)
option(ASAN "Wether to enable Address Sanitiazation with -fsanitize=address.")

# Permitted backend values
set(SPATTER_BACKENDS "serial" "openmp" "cuda")

# Permitted compiler values
set(SPATTER_COMPILERS "cray" "gnu" "clang" "armclang" "intel" "xl" "nvcc")

set(SPATTER_CPU_BACKENDS "serial" "openmp")
set(SPATTER_CPU_COMPILERS "cray" "gnu" "clang" "armclang" "intel" "xl")

# Check to ensure a backend is specified
if (NOT BACKEND)
    message(WARNING "No backend specified. Using serial CPU backend.")
    set(BACKEND "serial")
endif()

# Check for valid backend
if(NOT "${BACKEND}" IN_LIST SPATTER_BACKENDS)
    message(FATAL_ERROR "Backend must be one of ${SPATTER_BACKENDS}")
endif()

# Check to ensure a compiler is specified. Use sensible defaults.
if (NOT COMPILER)
    if (BACKEND IN_LIST SPATTER_CPU_BACKENDS)
        # If using a CPU backend, default to gcc
        message(WARNING "No compiler specified. Using gcc.")
        set(COMPILER "gnu")
    else(BACKEND STREQUAL "cuda")
        # Silently set compiler to nvcc for the cuda backend
        set(COMPILER "nvcc")
    endif()
endif()

#Check for valid compiler
if(NOT COMPILER IN_LIST SPATTER_COMPILERS)
    message(FATAL_ERROR "compiler must be one of ${SPATTER_COMPILERS}")
endif( )

#########################
# Create gz_read target #
#########################

# Create gz_read executable
add_executable(gz_read standard-suite/binary-traces/gz_read.c)
target_link_libraries(gz_read z)

#############
# Set flags #
#############

# Non-backend specific flags
if(ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif()

if (USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
endif()

# Backend specific flags

#SERIAL
if ("${BACKEND}" STREQUAL "serial")
    if ("${COMPILER}" STREQUAL "gnu")
        set(CMAKE_C_COMPILER gcc)
        set(CMAKE_CXX_COMPILER g++)
    elseif ("${COMPILER}" STREQUAL "cray")
        set(CMAKE_C_COMPILER cc)
        set(CMAKE_CXX_COMPILER CC)
    else ()
        message(FATAL_ERROR "Only gnu and cray are supported for SERIAL backend")
    endif ()
endif ()

#CUDA
if ("${BACKEND}" STREQUAL "cuda")
    if (DEFINED CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
    else ()
        message("No CUDA architecture specified, default set to 7.0")
        set(CMAKE_CUDA_ARCHITECTURES "70")
    endif ()
endif ()

#OPENMP
if ("${BACKEND}" STREQUAL "openmp")
    #gnu
    if ("${COMPILER}" STREQUAL "gnu")
        set(CMAKE_C_COMPILER gcc)
        set(CMAKE_CXX_COMPILER g++)

    #cray
    elseif ("${COMPILER}" STREQUAL "cray")
        set(CMAKE_C_COMPILER cc)
        set(CMAKE_CXX_COMPILER CC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=m")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hlist=m")
        set(CXX_STD 11)
        if (USE_SVE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -homp")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -homp")
            set(CMAKE_SYSTEM_NAME "Generic")
            set(CMAKE_PROCESSOR_NAME "arm")
        endif()

    #clang
    elseif ("${COMPILER}" STREQUAL "clang")
        set(CMAKE_C_COMPILER clang)
        set(CMAKE_CXX_COMPILER clang++)

    #armclang (wombat)
    elseif ("${COMPILER}" STREQUAL "armclang")
        set(CMAKE_CXX_COMPILER armclang)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=thunderx2t99 -mtune=thunderx2t99 -fopenmp")
        set(USE_OMP_SIMD 1)

    #xl
    elseif ("${COMPILER}" STREQUAL "xl")
        set(CMAKE_C_COMPILER xlc_r)
        set(CMAKE_CXX_COMPILER xlc++_r)

    #intel
    elseif ("${COMPILER}" STREQUAL "intel")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")

        #Check for intel platform variable set in user cmake call
        if (NOT DEFINED INTEL_PLATFORM)
            message("No intel platform defined. Enable more optimization by specifying -DINTEL_PLATFORM=[skylake;avx]")
        else()
            #Set platform
            ####
            string( TOLOWER "${INTEL_PLATFORM}" INTEL_PLATFORM )

            #skylake
            if("${INTEL_PLATFORM}" STREQUAL "skylake")
                set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -march=skylake-avx512")

            #cross-platform with AVX
            elseif("${INTEL_PLATFORM}" STREQUAL "avx")
                set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -xCOMMON-AVX512")

            else()
                message("Platform was not recognized, no options passed.")
            endif()
        endif()

        #Set compiler based on options
        set(CMAKE_C_COMPILER icx)
        set(CMAKE_CXX_COMPILER icpx)

        #Check disable cpu dispatch flag
        if (DISABLE_CPU_DISPATCH)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -diag-disable=15009")
        endif()
        if (VEC_REPORT)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -qopt-report-phase=vec,loop -qopt-report=2")
        endif()
    else()
        message (
            FATAL_ERROR
                "Only gnu, cray, clang, armclang, intel, and xl are supported for OpenMP backend"
        )
    endif()
endif ()

project (spatter C CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set target project name
set (TRGT spatter)

# Enable CUDA language
if ("${BACKEND}" STREQUAL "cuda")
    enable_language(CUDA)
endif ()

# Debug function to check all the variables in the CMakeFile
macro (print_all_variables)
    message (
        STATUS "print_all_variables------------------------------------------{"
    )
    get_cmake_property (_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message (STATUS "${_variableName}=${${_variableName}}")
    endforeach ()
    message (
        STATUS "print_all_variables------------------------------------------}"
    )
ENDMACRO ()

# Check that appropriate compiler versions are used
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message (FATAL_ERROR "GCC version must be at least 4.9.1 to run the OpenMP 4.0 backend")
endif ()

#Add specific flags for the CodeXL compiler
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
	set(IBMXL_COMPILE_FLAGS "-qenablevmx -qtune=pwr9")
	set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${IBMXL_COMPILE_FLAGS}")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=IBM")
endif ()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Cray")
    set (OPTIMIZATIONS "-O3 -h vector3 -h cache3 -h scalar3")
    #set (OPTIMIZATIONS "-G0 -O0")
    #set (ALLWARNINGS "-Wall -Wextra -ansi -pedantic")
    #set (ALLWARNINGS "-h msglevel_2")
    set (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OPTIMIZATIONS} ${ALLWARNINGS} -hlist=m -hlist=d ")
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPTIMIZATIONS} -hlist=m -D__CRAYC__")
    set (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Cray -D__CRAYC__")
endif ()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -xHost -qopenmp")
    set (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -xHost -qopenmp")
endif ()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=GNU -O3 -g")
   #Use sanitize=address for debugging of access issues
   #set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fsanitize=address")
endif ()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Clang")
endif ()

#Add configure-specified flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include -I${CMAKE_CURRENT_SOURCE_DIR}/external/argtable3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include -I${CMAKE_CURRENT_SOURCE_DIR}/external/argtable3")

#Check if Spatter is compiled on MacOSX. If so, use POSIX_MEMALIGN to allocate memory
if (APPLE)
	add_definitions (-DUSE_POSIX_MEMALIGN)
endif ()

# Enable OpenCL
if ("${BACKEND}" STREQUAL "opencl")
    add_definitions (-DUSE_OPENCL)
    # Point the compiler to the include and library directories
    include_directories ($ENV{OCL_INCL} src/opencl)
    # include_directories(/usr/lib/gcc/x86_64-linux-gnu/5/include/)
    link_directories ($ENV{OCL_LIB})
    # Pull the OpenCL-specific files into the build
    file (GLOB BACKEND_C_FILES_CL src/opencl/*.c)
    file (GLOB BACKEND_H_FILES_CL src/opencl/*.h)

    # Copy over the OpenCL kernels used with the binary
    file (GLOB OCL_KERNELS src/opencl/*.cl)
    file (
        COPY
        ${OCL_KERNELS}
        include/sgtype.h
        DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/
    )

    message ("Using OpenCL backend")

endif ()

# Enable Serial backend
if ("${BACKEND}" STREQUAL "serial")
    add_definitions (-DUSE_SERIAL)
    include_directories (src/serial)
    # Pull the serial files and kernels into the build
    file (GLOB BACKEND_C_FILES_SERIAL src/serial/*.c)
    file (GLOB BACKEND_H_FILES_SERIAL src/serial/*.h)
    message ("Using serial backend")
endif ()

#Enable MPI
if (USE_MPI)
    add_definitions (-DUSE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
endif()

# Enable OpenMP
if ("${BACKEND}" STREQUAL "openmp")
    find_package (OpenMP REQUIRED)
    if(OpenMP_FOUND)
	# enable pragma regions in the code for OpenMP
        add_definitions (-DUSE_OPENMP)
        add_definitions (-DUSE_OMP_SIMD)
        include_directories (src/openmp)

        # Pull the OpenMP-specific files and kernels into the build
        file (GLOB BACKEND_C_FILES_OMP src/openmp/*.c)
        file (GLOB BACKEND_H_FILES_OMP src/openmp/*.h)
    endif ()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message ("Using OpenMP backend")
endif ()

# Enable CUDA
if ("${BACKEND}" STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    add_definitions (-DUSE_CUDA)
    include_directories (src/cuda)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    set (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )

    file (GLOB CUDA_CU_FILES src/cuda/*.cu)
    file (GLOB CUDA_C_FILES src/cuda/*.c)
    file (GLOB CUDA_H_FILES src/cuda/*.h)

    #CUDA Toolkit (Runtime)
    add_library(cuda_comp SHARED src/cuda/cuda-backend.cu)
    #add_library(cuda_comp SHARED src/cuda/my_kernel.cu src/cuda/cuda-backend.cu src/cuda/cuda-backend.h)
    set_target_properties(cuda_comp
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
    )
    target_include_directories(cuda_comp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

    message ("Using CUDA backend")

endif ()

if ("${BACKEND}" STREQUAL "cuda_jit")
    find_package(CUDAToolkit REQUIRED)
    add_definitions (-DUSE_CUDA_JIT)
    include_directories (src/cuda_jit)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    set (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )

    file (GLOB CUDA_CU_FILES src/cuda_jit/*.cu)
    file (GLOB CUDA_C_FILES src/cuda_jit/*.c)
    file (GLOB CUDA_H_FILES src/cuda_jit/*.h)

    #CUDA Toolkit (Runtime)
    add_library(cuda_comp SHARED src/cuda_jit/cuda-jit-backend.cu)
    #add_library(cuda_comp SHARED src/cuda/my_kernel.cu src/cuda/cuda-backend.cu src/cuda/cuda-backend.h)
    set_target_properties(cuda_comp
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
    )
    target_include_directories(cuda_comp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_jit")


    message ("Using CUDA_JIT backend")
endif ()

if (USE_PAPI)
    set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    include_directories (src/papi)
    add_definitions (-DUSE_PAPI)
    #Use the included FindPAPI cmake module
    include (FindPAPI)
    find_package (PAPI REQUIRED)
    #message("Found PAPI libraries at ${PAPI_INCLUDE_DIRS}")
    include_directories(${PAPI_INCLUDE_DIRS})
    file (GLOB PAPI_C_FILES src/papi/*.c)
    file (GLOB PAPI_H_FILES src/papi/*.h)
endif ()

# Include the location of stddef.h include_directories(/usr/include/linux/)

# Include amalgamated argtable files
file (GLOB EXTERNAL_C_FILES external/argtable3/*.c)
file (GLOB EXTERNAL_H_FILES external/argtable3/*.h)

# Glob for the source files and headers in the current directory and kernels
file (GLOB C_FILES src/*.c)
file (GLOB H_FILES src/include/*.h)
set (
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
add_executable (${TRGT} ${SOURCE_FILES})

#Prefer C11 standard to allow for loop initialization and aligned_alloc
target_compile_features(${TRGT} PUBLIC c_std_11)

# Print out debug info print_all_variables()

# Link with the appropriate libraries
if ("${BACKEND}" STREQUAL "openmp")
    target_link_libraries (${TRGT} LINK_PUBLIC OpenMP::OpenMP_CXX)
endif()

if ("${BACKEND}" STREQUAL "opencl")
    target_link_libraries (${TRGT} LINK_PUBLIC OpenCL)
endif ()

# Link math library for json
target_link_libraries (${TRGT} LINK_PUBLIC m)

if ("${BACKEND}" STREQUAL "cuda")
    target_link_libraries (${TRGT} PUBLIC cuda_comp)
    target_link_libraries (${TRGT} PUBLIC CUDA::cudart)
endif ()

if ("${BACKEND}" STREQUAL "cuda_jit")
    target_link_libraries (${TRGT} PUBLIC cuda_comp)
    target_link_libraries (${TRGT} PUBLIC CUDA::cudart)
endif ()

#Include PAPI libraries, if defined
if (USE_PAPI)
    target_link_libraries (${TRGT} LINK_PUBLIC ${PAPI_LIBRARIES})
    # Explicitly link libdl because the Cray compiler refuses to link papi dynaimically...
    # (Static libraries carry no dependency info, and libpapi depends on libdl)
    target_link_libraries (${TRGT} LINK_PUBLIC dl)
endif ()

#Link MPI libraries
if (USE_MPI)
    target_link_libraries(${TRGT} PUBLIC MPI::MPI_CXX)
endif()

# Copy over the test scripts
file (GLOB TEST_SCRIPTS tests/*.sh)
file (COPY ${TEST_SCRIPTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Add a command to put some important info in environment variables
# to be output by spatter
message (STATUS "compiler version is ${CMAKE_C_COMPILER_VERSION}")
message (STATUS "compiler is ${CMAKE_C_COMPILER}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C=\"${CMAKE_C_COMPILER}\"")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C_VER=\"${CMAKE_C_COMPILER_VERSION}\"")

# Build tests
if ( "${BACKEND}" STREQUAL "openmp" OR "${BACKEND}" STREQUAL "serial" OR "${BACKEND}" STREQUAL "cuda")
    mark_as_advanced( BUILD_TESTS )
    set( BUILD_TESTS true CACHE BOOL "Tests build target available if true" )
    if( BUILD_TESTS )
        enable_testing()
        add_subdirectory( tests )
    endif()
endif()

# Validation flag
set (VALIDATE_DATA 0 CACHE BOOL "Performs additional validation")
if ("${BACKEND}" STREQUAL "openmp" OR "${BACKEND}" STREQUAL "cuda")
    if (VALIDATE_DATA)
        add_definitions(-DVALIDATE)
    endif ()
endif ()


# Note - This file can be reformatted with cmake-format CMakeLists.txt -i
# --command-case=upper --keyword-case=upper --dangle-parens --tab-size=4
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do not
# have cmake-format installed, install it with pip install --user cmake_format
