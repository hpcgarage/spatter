# 3.18 is specified to include FindCUDAToolkit support updates
cmake_minimum_required (VERSION 3.18 FATAL_ERROR)
project (spatter C CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set target project name
set (TRGT spatter)

# Use for debugging. TODO: Set this automatically
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Use ASAN when debugging
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# Check for support for at least one backend
if ("${BACKEND}" STREQUAL "")
    message (
        FATAL_ERROR
            "You must build with support for at least one backend. Pass at least one of -DBACKEND=serial, openmp, or cuda to cmake."
    )
endif ()

# Check for support for at least one compiler
if ("${COMPILER}" STREQUAL "")
    message (	
        FATAL_ERROR	
            "You must build with support for at least one compiler. Pass at least one of -DCOMPILER=cray, gnu, intel, xl, or nvidia to cmake."	
    )	
endif()

#Make backend and compiler variables case insensitive
string( TOLOWER "${BACKEND}" BACKEND )
string( TOLOWER "${COMPILER}" COMPILER )

#Set backend permitted values
set(SPATTERBACKENDS serial openmp cuda)

#Check for backend variable set in user cmake call
if(NOT BACKEND IN_LIST SPATTERBACKENDS)
    message(FATAL_ERROR "backend must be one of ${SPATTERBACKENDS}")
endif()

#Set compiler permitted values
set(SPATTERCOMPILERS cray gnu clang armclang intel xl nvcc)

#Set intel platform permitted values
set(INTELPLATFORMS skylake avx_crossplatform non_avx)

#Check for compiler variable set in user cmake call
if(NOT COMPILER IN_LIST SPATTERCOMPILERS)
    message(FATAL_ERROR "compiler must be one of ${compilerBackends}")
endif()

add_definitions (-DSG_DATA_TYPE=double)	

# Set backend specific flags	
    #SERIAL	
    if ("${BACKEND}" STREQUAL "serial")	
        #gnu
        if ("${COMPILER}" STREQUAL "gnu")
            set(CMAKE_C_COMPILER gcc)
            set(CMAKE_CXX_COMPILER g++)
        #cray
        elseif ("${COMPILER}" STREQUAL "cray")
            set(CMAKE_C_COMPILER cc)
            set(CMAKE_CXX_COMPILER CC)
        else()
            message (	
                FATAL_ERROR
                    "Only gnu and cray are supported for SERIAL backend"	
            )
        endif()
    endif ()

    #CUDA	
    if ("${BACKEND}" STREQUAL "cuda")	
        if ("${COMPILER}" STREQUAL "nvcc")
            set(CMAKE_CXX_COMPILER nvcc)
            #TODO: How will user specify CUDA arch	
            set(CMAKE_CUDA_ARCHITECTURES "70")
        else()
            message (	
                FATAL_ERROR
                    "Only nvcc is supported for CUDA backend"
            )
        endif()
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
            if (USE_SVE)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -homp -hlist=m")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static -homp -hlist=m")
                set(CXX_STD 11)
                set(CMAKE_SYSTEM_NAME "Generic")
                set(CMAKE_PROCESSOR_NAME "arm")
            elseif (USE_PAPI)
                set(PAPI_DIR "/opt/cray/pe/papi/default/")
            else()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=m")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hlist=m")
                set(CXX_STD 11)
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
                message("No intel platform defined.")
            else()
                #Set platform
                ####
                string( TOLOWER "${INTEL_PLATFORM}" INTEL_PLATFORM )

                #skylake
                if("${INTEL_PLATFORM}" STREQUAL "skylake")
                    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -march=skylake-avx512")

                #cross-platform with AVX
                elseif("${INTEL_PLATFORM}" STREQUAL "avx_crossplatform")
                    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -xCOMMON-AVX512")
                
                #Non-AVX platforms
                elseif("${INTEL_PLATFORM}" STREQUAL "non_avx")
                    #No specified flags
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
IF (USE_MPI)
    ADD_DEFINITIONS (-DUSE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
ENDIF()

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
    add_definitions (-DUSE_CUDA)
    include_directories (src/cuda)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    set (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )
    if ("${BACKEND}" STREQUAL "openmp")
        set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif ()

    file (GLOB CUDA_CU_FILES src/cuda/*.cu)
    file (GLOB CUDA_C_FILES src/cuda/*.c)
    file (GLOB CUDA_H_FILES src/cuda/*.h)

    #CUDA Toolkit (Runtime)
    add_library(cuda_comp SHARED src/cuda/my_kernel.cu src/cuda/cuda-backend.cu src/cuda/cuda-backend.h src/cuda/cuda_kernels.h)
    set_target_properties(cuda_comp
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
    )
    target_include_directories(cuda_comp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

    message ("Using CUDA backend")

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
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do NOT
# have cmake-format installed, install it with pip install --user cmake_format