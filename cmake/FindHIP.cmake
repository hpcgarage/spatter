# cmake/FindHIP.cmake
if(NOT DEFINED HIP_PATH)
  message(STATUS "HIP_PATH not defined; trying to locate hipcc in PATH.")
  find_program(HIPCC_EXECUTABLE NAMES hipcc)
else()
  set(HIPCC_EXECUTABLE "${HIP_PATH}/bin/hipcc")
endif()

if(NOT HIPCC_EXECUTABLE)
  message(FATAL_ERROR "hipcc not found. Please ensure HIP is installed and/or set HIP_PATH to the HIP installation root.")
endif()

get_filename_component(HIP_BIN_DIR "${HIPCC_EXECUTABLE}" DIRECTORY)
get_filename_component(HIP_ROOT "${HIP_BIN_DIR}" DIRECTORY)

set(HIP_INCLUDE_DIRS "${HIP_ROOT}/include")
set(HIP_LIBRARIES "hip")

if(NOT EXISTS "${HIP_INCLUDE_DIRS}/hip/hip_runtime.h")
    message(WARNING "Expected HIP header not found at ${HIP_INCLUDE_DIRS}/hip/hip_runtime.h. "
                    "If hipcc is in /usr/bin, then HIP_ROOT is set to ${HIP_ROOT}. "
                    "This may not be the actual HIP installation directory. "
                    "Consider setting -DHIP_PATH to the correct HIP installation root.")
endif()

set(HIP_FOUND TRUE)
set(HIP_CONFIG_VERSION "0.0.1")

message(STATUS "Found HIP installation at: ${HIP_ROOT}")
message(STATUS "HIP include directories: ${HIP_INCLUDE_DIRS}")
message(STATUS "HIP libraries: ${HIP_LIBRARIES}")

mark_as_advanced(HIP_INCLUDE_DIRS HIP_LIBRARIES HIP_ROOT)

