# Spatter/src/Spatter/CMakeLists.txt
set(SUFFIX "")
if (USE_CUDA)
    add_library(cuda_backend SHARED CudaBackend.cu)
    set_target_properties(cuda_backend PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    set(BACKEND_INCLUDE_FILES CudaBackend.hh)
    set(BACKEND_LIBRARY cuda_backend)
elseif (USE_HIP)
    add_library(hip_backend SHARED HipBackend.cpp)
    set_source_files_properties(HipBackend.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(hip_backend PRIVATE -x hip)
    set(BACKEND_INCLUDE_FILES HipBackend.hh)
    set(BACKEND_LIBRARY hip_backend)
    set(SUFFIX ".hip")
endif()

set(SPATTER_INCLUDE_FILES
    ${BACKEND_INCLUDE_FILES}
    Configuration${SUFFIX}.hh
    Input${SUFFIX}.hh
    JSONParser${SUFFIX}.hh
    PatternParser${SUFFIX}.hh
    SpatterTypes${SUFFIX}.hh
    AlignedAllocator${SUFFIX}.hh
    Timer.hh
)

add_library(Spatter STATIC
    ${SPATTER_INCLUDE_FILES}
    Configuration${SUFFIX}.cc
    JSONParser${SUFFIX}.cc
    PatternParser${SUFFIX}.cc
    Timer${SUFFIX}.cc
)

add_library(Spatter_shared SHARED
    ${SPATTER_INCLUDE_FILES}
    Configuration${SUFFIX}.cc
    JSONParser${SUFFIX}.cc
    PatternParser${SUFFIX}.cc
    Timer${SUFFIX}.cc
)

set_target_properties(Spatter PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${COMMON_COMPILE_OPTIONS}"
    OUTPUT_NAME "Spatter"
)

set_target_properties(Spatter_shared PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${COMMON_COMPILE_OPTIONS}"
    OUTPUT_NAME "Spatter"
)

target_include_directories(Spatter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/Spatter>
)

target_include_directories(Spatter_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/Spatter>
)

if (USE_CUDA OR USE_HIP)
    set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${BACKEND_LIBRARY})
endif()

target_link_libraries(Spatter PUBLIC ${COMMON_LINK_LIBRARIES})
target_link_libraries(Spatter_shared PUBLIC ${COMMON_LINK_LIBRARIES})

target_compile_options(Spatter PRIVATE ${WARNING_FLAGS})
target_compile_options(Spatter_shared PRIVATE ${WARNING_FLAGS})
