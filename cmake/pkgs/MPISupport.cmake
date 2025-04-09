option(USE_MPI "Enable support for MPI")

if (USE_MPI)
    find_package(MPI)
    include_directories(${MPI_INCLUDE_PATH})
    #Explicitly add directory for Ubuntu 22 to search
    include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
    set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} MPI::MPI_CXX)
    add_definitions(-DUSE_MPI)
endif()
