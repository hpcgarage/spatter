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

set (SPAT_CXX_VER ${CMAKE_CXX_COMPILER_VERSION})

add_definitions(-DSPAT_CXX_NAME=${SPAT_CXX_NAME})
add_definitions(-DSPAT_CXX_VER=${SPAT_CXX_VER})

message(STATUS "Setting SPAT_CXX_NAME to '${SPAT_CXX_NAME}'")
message(STATUS "Setting SPAT_CXX_VER to '${SPAT_CXX_VER}'")
