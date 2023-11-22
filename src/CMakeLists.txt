# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter-driver main.cc)

target_link_libraries(spatter-driver ${COMMON_LINK_LIBRARIES} Spatter)
target_include_directories(spatter-driver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

set_target_properties(spatter-driver PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${WARNING_FLAGS}"
    )
