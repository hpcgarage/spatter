option(USE_HIP "Enable support for HIP" ON)

if (USE_HIP)
    # Attempt to find HIP. (This should use your FindHIP.cmake.)
    find_package(HIP REQUIRED)
    if (HIP_FOUND)
        # Set the C++ standard for HIP builds.
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        
        # Define a preprocessor macro so your code can conditionally compile HIP-specific code.
        add_definitions(-DUSE_HIP)
        
        # Add the HIP include directories so that HIP headers can be found.
        include_directories(${HIP_INCLUDE_DIRS})
        
        # Append HIP libraries to a common variable that you can use when linking executables.
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${HIP_LIBRARIES})
        
        message(STATUS "HIP support enabled.")
    else()
        message(STATUS "No HIP installation found.")
    endif()
endif()

