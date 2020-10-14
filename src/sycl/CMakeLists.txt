CMAKE_MINIMUM_REQUIRED (VERSION 3.0)
PROJECT (sycl_kernels)

# Set SYCL compiler flags and definitions
ADD_DEFINITIONS (-DUSE_SYCL)
IF (USE_FPGA_EMULATOR)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")
    ADD_DEFINITIONS (-DFPGA_EMULATOR)
ELSEIF (USE_FPGA_REPORT)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -fsycl-link -Xshardware")
    ADD_DEFINITIONS (-DFPGA_EMULATOR)
ELSEIF (USE_FPGA_BITSTREAM)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -Xshardware")
ELSEIF (USE_FPGA_PROFILING)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga -Xshardware -Xsprofile")
ELSE () #Use CPU emulation by default
    message("Building for CPU host emulation")
    ADD_DEFINITIONS (-DCPU_HOST)
ENDIF ()

ADD_LIBRARY (sycl_kernels_shared SHARED ${PROJECT_SOURCE_DIR}/sycl_kernel.cpp)

TARGET_LINK_LIBRARIES (sycl_kernels_shared LINK_PUBLIC sycl)
