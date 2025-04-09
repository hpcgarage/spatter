# Spatter/src/CMakeLists.txt
add_subdirectory(Spatter)
add_executable(spatter main.cc)

if(USE_HIP)
    target_compile_definitions(spatter PRIVATE USE_HIP)
endif()
if(USE_CUDA)
    target_compile_definitions(spatter PRIVATE USE_CUDA)
endif()

target_link_libraries(spatter ${COMMON_LINK_LIBRARIES} Spatter)

set_target_properties(spatter PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${WARNING_FLAGS}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
