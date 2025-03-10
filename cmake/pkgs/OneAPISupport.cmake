# Option to enable OneAPI
option(USE_ONEAPI "Enable support for Intel OneAPI" ON)

if (USE_ONEAPI)
    # Check for compiler and print debug info
    include(CheckLanguage)
    check_language(CXX)

    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

    # Explicitly check if the compiler is IntelLLVM (DPC++)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        message(STATUS "OneAPI DPC++ compiler (IntelLLVM) detected: ${CMAKE_CXX_COMPILER}")

        # Set compiler flags
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        # Define a preprocessor directive for OneAPI support
        add_definitions(-DUSE_ONEAPI)

        # First try finding IntelSYCL (New recommended package)
        set(IntelSYCL_DIR "/net/projects/tools/x86_64/rhel-8/intel-oneapi/2024.2/compiler/latest/lib/cmake/sycl")
        find_package(IntelSYCL REQUIRED)

        if (IntelSYCL_FOUND)
            message(STATUS "Intel OneAPI SYCL package found.")
            set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} IntelSYCL::SYCL_CXX)
        else()
            message(WARNING "IntelSYCL::SYCL target not found! Falling back to manual linking.")

            # Manually link the Intel SYCL library
            set(SYCL_LIB_PATH "/net/projects/tools/x86_64/rhel-8/intel-oneapi/2024.2/compiler/2024.2/lib")

            # Correct OneAPI include path
            set(SYCL_INCLUDE_PATH "/net/projects/tools/x86_64/rhel-8/intel-oneapi/2024.2/compiler/2024.2/include")

            include_directories(${SYCL_INCLUDE_PATH})
            link_directories(${SYCL_LIB_PATH})

            set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} "${SYCL_LIB_PATH}/libsycl.so")
        endif()

        # Force CMake to use DPC++ compiler
        set(CMAKE_CXX_COMPILER icpx)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${SYCL_INCLUDE_PATH}")

    else()
        message(FATAL_ERROR "OneAPI DPC++ compiler not found. Detected CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()
