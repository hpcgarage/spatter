option(USE_MPI "Enable support for MPI")

if (USE_MPI)
    find_package(MPI)
    include_directories(${MPI_INCLUDE_PATH})
    set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} MPI::MPI_CXX)
endif()
