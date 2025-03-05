# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter main.cc)

# Ensure OneAPI library is linked correctly
if (USE_ONEAPI)
    # target_compile_options(spatter PUBLIC -fsycl)
    target_compile_options(spatter PUBLIC -fsycl -I/net/projects/tools/x86_64/rhel-8/intel-oneapi/2024.2/dpcpp-ct/2024.2/include -I/net/projects/tools/x86_64/rhel-8/intel-oneapi/2024.2/2024.2/include)
    target_link_options(spatter PUBLIC -fsycl)
endif()

target_link_libraries(spatter ${COMMON_LINK_LIBRARIES} Spatter)

set_target_properties(spatter PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${WARNING_FLAGS}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )