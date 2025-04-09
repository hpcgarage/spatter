option(USE_OPENMP "Enable support for OpenMP")

if (USE_OPENMP)
    find_package(OpenMP)
    set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} OpenMP::OpenMP_CXX)
    add_definitions(-DUSE_OPENMP)
endif()
