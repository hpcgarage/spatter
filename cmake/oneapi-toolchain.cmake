# Pass this file to the first invocation of cmake using -DCMAKE_TOOLCHAIN_FILE=
set(CMAKE_SYSTEM_NAME Linux)

#Usually ONEAPI_ROOT will be set by a call to setvars.sh
if(DEFINED ENV{ONEAPI_ROOT})
	set(ONEAPI_BASE $ENV{ONEAPI_ROOT})
else()
	set(ONEAPI_BASE "/opt/intel/inteloneapi")
endif()

message(STATUS "Using DPCPP toolchain in ${ONEAPI_BASE}")

set(CMAKE_CXX_COMPILER "${ONEAPI_BASE}/compiler/latest/linux/bin/dpcpp")
