option(USE_ONEAPI "Enable support for OneApi")

if (USE_ONEAPI)
    add_definitions(-DUSE_ONEAPI)
endif()
