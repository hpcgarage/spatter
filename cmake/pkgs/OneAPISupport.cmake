# OneAPISupport.cmake
#
# IMPORTANT: The Intel DPC++/C++ compiler (icpx) must be selected BEFORE
# CMake's project() call.  Pass it on the command line:
#   cmake -DCMAKE_CXX_COMPILER=icpx ...
# or source Intel's environment script first:
#   source /opt/intel/oneapi/setvars.sh && cmake -DCMAKE_CXX_COMPILER=icpx ...

if (USE_ONEAPI)
    # Verify that icpx (IntelLLVM) is actually the active CXX compiler.
    # CMAKE_CXX_COMPILER cannot be changed after project() has been called.
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        message(FATAL_ERROR
            "USE_ONEAPI requires the Intel DPC++/C++ compiler (icpx/icx).\n"
            "  Detected: CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            " (${CMAKE_CXX_COMPILER})\n"
            "  Reconfigure with: -DCMAKE_CXX_COMPILER=icpx\n"
            "  or source Intel's setvars.sh before running cmake.")
    endif()

    message(STATUS "OneAPI DPC++ compiler (IntelLLVM) detected: ${CMAKE_CXX_COMPILER}")

    # -fsycl is required for SYCL device compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    add_definitions(-DUSE_ONEAPI)

    # ------------------------------------------------------------------
    # Locate the IntelSYCL CMake package.
    # Search order:
    #   1. User-supplied IntelSYCL_DIR cache variable
    #   2. CMPLR_ROOT environment variable  (set by setvars.sh / modulefiles)
    #   3. ONEAPI_ROOT environment variable (set by setvars.sh)
    #   4. Derive from the compiler executable location
    # ------------------------------------------------------------------
    if (NOT DEFINED IntelSYCL_DIR)
        if (DEFINED ENV{CMPLR_ROOT})
            set(IntelSYCL_DIR "$ENV{CMPLR_ROOT}/lib/cmake/sycl"
                CACHE PATH "Path to IntelSYCL CMake config")
        elseif (DEFINED ENV{ONEAPI_ROOT})
            set(IntelSYCL_DIR "$ENV{ONEAPI_ROOT}/compiler/latest/lib/cmake/sycl"
                CACHE PATH "Path to IntelSYCL CMake config")
        else()
            # Fall back to a path relative to the compiler binary
            get_filename_component(_icpx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set(IntelSYCL_DIR "${_icpx_bindir}/../lib/cmake/sycl"
                CACHE PATH "Path to IntelSYCL CMake config")
        endif()
    endif()

    find_package(IntelSYCL QUIET)

    if (IntelSYCL_FOUND)
        message(STATUS "Intel OneAPI SYCL package found (${IntelSYCL_DIR})")
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} IntelSYCL::SYCL_CXX)
    else()
        message(WARNING
            "IntelSYCL CMake package not found (searched: ${IntelSYCL_DIR}).\n"
            "Falling back to manual SYCL library linking.")

        # Derive SYCL lib/include paths with the same priority order
        if (DEFINED ENV{CMPLR_ROOT})
            set(_sycl_root "$ENV{CMPLR_ROOT}")
        elseif (DEFINED ENV{ONEAPI_ROOT})
            set(_sycl_root "$ENV{ONEAPI_ROOT}/compiler/latest")
        else()
            get_filename_component(_icpx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set(_sycl_root "${_icpx_bindir}/..")
        endif()

        set(SYCL_LIB_PATH     "${_sycl_root}/lib")
        set(SYCL_INCLUDE_PATH "${_sycl_root}/include")

        if (NOT EXISTS "${SYCL_LIB_PATH}/libsycl.so")
            message(FATAL_ERROR
                "Could not find libsycl.so under ${SYCL_LIB_PATH}.\n"
                "Set CMPLR_ROOT or ONEAPI_ROOT, or specify -DIntelSYCL_DIR=<path>.")
        endif()

        include_directories(${SYCL_INCLUDE_PATH})
        link_directories(${SYCL_LIB_PATH})
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} "${SYCL_LIB_PATH}/libsycl.so")
        message(STATUS "Using manual SYCL library: ${SYCL_LIB_PATH}/libsycl.so")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${SYCL_INCLUDE_PATH}")
    endif()

    message(STATUS "OneAPI support enabled (flags: ${CMAKE_CXX_FLAGS})")
endif()
