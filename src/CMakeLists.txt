# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter-driver main.cc)
#target_compile_options(spatter-driver PUBLIC "-fnew-alignment 32")
target_link_libraries(spatter-driver ${COMMON_LINK_LIBRARIES} Spatter)
set_target_properties(spatter-driver PROPERTIES
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${WARNING_FLAGS}"
    )
