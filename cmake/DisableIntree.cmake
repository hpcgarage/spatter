# Check for in-tree builds.  This must appear before project()
if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  message(FATAL_ERROR "In-tree builds are disabled: please configure in \
                       another subdirectory")
endif()

