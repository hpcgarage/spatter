# Spatter/src/Spatter/CMakeLists.txt

configure_file(Spatter_config.hpp.cmakein Spatter_config.hpp)

if (USE_CUDA)
    add_library(cuda_backend SHARED CudaBackend.cu)
    set_target_properties(cuda_backend PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    set(CUDA_INCLUDE_FILES CudaBackend.hh)
endif()

set(SPATTER_INCLUDE_FILES
    ${CUDA_INCLUDE_FILES}
    Configuration.hh
    Input.hh
    SpatterTypes.hh
    Timer.hh)

add_library(Spatter INTERFACE)

set_target_properties(Spatter PROPERTIES
    INTERFACE_COMPILE_FEATURES cxx_std_20
    COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
    COMPILE_OPTIONS "${COMMON_COMPILE_OPTIONS}"
    OUTPUT_NAME "Spatter"
    )

target_include_directories(Spatter
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (USE_CUDA)
    set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} cuda_backend)
endif()

target_link_libraries(Spatter
    INTERFACE
    ${COMMON_LINK_LIBRARIES}
    )

target_compile_options(Spatter
    INTERFACE ${WARNING_FLAGS}
    )

#install(TARGETS Spatter
#    EXPORT Spatter_Targets
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    )

#install(EXPORT Spatter_Targets
#    FILE Spatter_Targets.cmake
#    NAMESPACE Spatter::
#    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Spatter
#)

#install(FILES ${SPATTER_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Spatter_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

