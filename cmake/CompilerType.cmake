#Add specific flags for the CodeXL compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(IBMXL_COMPILE_FLAGS "-qenablevmx -qtune=pwr9")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${IBMXL_COMPILE_FLAGS}")
    set(SPAT_CXX_NAME "IBM")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray") 
    set (OPTIMIZATIONS "-O3 -h vector3 -h cache3 -h scalar3")
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPTIMIZATIONS} -hlist=m -D__CRAYC__")
    set(SPAT_CXX_NAME "Cray")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Intel)(LLVM)?")
    set (OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -xHost -qopenmp")
    set(SPAT_CXX_NAME "Intel")
endif ()

 
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") 
    set(SPAT_CXX_NAME "GNU")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") 
    set(SPAT_CXX_NAME "Clang")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(SPAT_CXX_NAME "AppleClang")
endif ()

# ---------------------------------------------------------------------------
# Host-architecture optimization for GNU/Clang.
#
# Spatter is a memory microbenchmark, so the compiler must target the ISA of
# the machine actually being measured. The Intel compiler above already does
# this with -xHost, but with GNU/Clang no architecture flag was being added,
# so the compiler defaulted to the baseline x86-64 (SSE2) ISA and never
# emitted AVX/AVX2/AVX-512 instructions in the gather/scatter kernels
# (see issue #209).
#
# SPATTER_ENABLE_NATIVE_ARCH (default ON) adds the GNU/Clang equivalent of
# -xHost. For reproducible or cross-compiled builds, either turn it OFF or
# set SPATTER_ARCH_FLAGS to pin a specific target, e.g.
#   -DSPATTER_ARCH_FLAGS="-march=sapphirerapids"
# When SPATTER_ARCH_FLAGS is set it takes precedence and native detection is
# skipped. Native tuning is also skipped automatically when cross-compiling.
# ---------------------------------------------------------------------------
option(SPATTER_ENABLE_NATIVE_ARCH "Tune the build for the host CPU architecture (GNU/Clang)" ON)
set(SPATTER_ARCH_FLAGS "" CACHE STRING "Explicit architecture flags for GNU/Clang (overrides native detection), e.g. -march=native or -march=sapphirerapids")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(GNU|Clang|AppleClang)$")
    if (SPATTER_ARCH_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPATTER_ARCH_FLAGS}")
        message(STATUS "Using user-specified SPATTER_ARCH_FLAGS: ${SPATTER_ARCH_FLAGS}")
    elseif (SPATTER_ENABLE_NATIVE_ARCH AND NOT CMAKE_CROSSCOMPILING)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" SPATTER_HAS_MARCH_NATIVE)
        if (SPATTER_HAS_MARCH_NATIVE)
            set(SPATTER_NATIVE_FLAG "-march=native")
        else ()
            # Apple/arm64 toolchains generally accept -mcpu=native instead.
            check_cxx_compiler_flag("-mcpu=native" SPATTER_HAS_MCPU_NATIVE)
            if (SPATTER_HAS_MCPU_NATIVE)
                set(SPATTER_NATIVE_FLAG "-mcpu=native")
            endif ()
        endif ()
        if (SPATTER_NATIVE_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPATTER_NATIVE_FLAG}")
            message(STATUS "Enabling host-native architecture tuning: ${SPATTER_NATIVE_FLAG}")
        else ()
            message(STATUS "No host-native architecture flag supported by this compiler; skipping")
        endif ()
    else ()
        message(STATUS "Host-native architecture tuning disabled (SPATTER_ENABLE_NATIVE_ARCH=${SPATTER_ENABLE_NATIVE_ARCH}, CROSSCOMPILING=${CMAKE_CROSSCOMPILING})")
    endif ()
endif ()

set (SPAT_CXX_VER ${CMAKE_CXX_COMPILER_VERSION})

add_definitions(-DSPAT_CXX_NAME=${SPAT_CXX_NAME})
add_definitions(-DSPAT_CXX_VER=${SPAT_CXX_VER})

message(STATUS "Setting SPAT_CXX_NAME to '${SPAT_CXX_NAME}'")
message(STATUS "Setting SPAT_CXX_VER to '${SPAT_CXX_VER}'")
