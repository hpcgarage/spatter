option(USE_HIP "Enable support for HIP" ON)

if (USE_HIP)
    if(NOT DEFINED HIP_PATH)
        set(HIP_PATH "/opt/rocm-6.3.1")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake")
    
    find_package(hip REQUIRED CONFIG)
    
    if (hip_FOUND)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        
        add_definitions(-DUSE_HIP)
        
        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES}
            hip::device
            amdhip64
        )
        
        message(STATUS "HIP support enabled.")
        message(STATUS "HIP path: ${HIP_PATH}")
        message(STATUS "HIP version: ${hip_VERSION}")
    else()
        message(FATAL_ERROR "HIP not found")
    endif()
else()
    message(STATUS "HIP support disabled")
endif()
