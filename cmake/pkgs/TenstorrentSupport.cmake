# Tenstorrent device kernels are compiled at run time by tt-metal's vendored
# sfpi toolchain, so there is no device compiler to enable here; only the host
# library matters at build time. There is no upstream FindTTMetal module.

option(USE_TENSTORRENT "Enable support for Tenstorrent accelerators")

if (USE_TENSTORRENT)
    set(TT_METAL_INCLUDE_DIRS "" CACHE STRING
        "Semicolon-separated include roots for tt-metal host headers")
    set(TT_METAL_LIB_DIR "" CACHE PATH "Directory containing libtt_metal.so")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                    "import ttnn, os; print(os.path.dirname(ttnn.__file__))"
            OUTPUT_VARIABLE TTNN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if (TTNN_DIR)
            message(STATUS "Tenstorrent: found ttnn wheel at ${TTNN_DIR}")
        endif()
    endif()

    find_library(TT_METAL_LIB
        NAMES tt_metal
        HINTS ${TT_METAL_LIB_DIR}
              ${TTNN_DIR}/build/lib
              $ENV{TT_METAL_HOME}/build/lib
              $ENV{TT_METAL_RUNTIME_ROOT}/build/lib)

    # Headers include each other as <tt-metalium/...>, so the root is the parent
    # of tt-metalium/.
    find_path(TT_METALIUM_INCLUDE_DIR
        NAMES tt-metalium/host_api.hpp
        HINTS ${TT_METAL_INCLUDE_DIRS}
              $ENV{TT_METAL_HOME}/tt_metal/api
              ${TTNN_DIR}/tt_metal/api)

    if (TT_METAL_LIB AND TT_METALIUM_INCLUDE_DIR)
        message(STATUS "Found libtt_metal: ${TT_METAL_LIB}")
        message(STATUS "Found tt-metalium headers: ${TT_METALIUM_INCLUDE_DIR}")

        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        include_directories(${TT_METALIUM_INCLUDE_DIR})
        if (TT_METAL_INCLUDE_DIRS)
            include_directories(${TT_METAL_INCLUDE_DIRS})
        endif()

        set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${TT_METAL_LIB})
        add_definitions(-DUSE_TENSTORRENT)
    else()
        if (TT_METAL_LIB AND NOT TT_METALIUM_INCLUDE_DIR)
            message(STATUS
                "Tenstorrent: libtt_metal was found but the host headers were not. "
                "A pip `ttnn` wheel ships the library and the device-kernel headers "
                "but no host API headers. Point -DTT_METAL_INCLUDE_DIRS at a matching "
                "tt-metal checkout plus its header-only dependencies, e.g.\n"
                "  git clone --depth 1 --branch v0.72.0 --filter=blob:none --sparse \\\n"
                "    https://github.com/tenstorrent/tt-metal ~/ttmetal-src\n"
                "  cd ~/ttmetal-src && git sparse-checkout set tt_metal tt_stl\n"
                "  git submodule update --init --depth 1 tt_metal/third_party/umd\n"
                "then clone fmt, nlohmann/json, tt-logger, spdlog and enchantum at the "
                "versions pinned in that checkout's third_party/CMakeLists.txt and pass "
                "every include root. Do NOT run tt-metal's own configure for this: it "
                "pulls system packages (boost, capnproto, protobuf) that need root, "
                "whereas the header-only clones do not.")
        elseif (NOT TT_METAL_LIB)
            message(STATUS
                "Tenstorrent: libtt_metal not found. Install tt-metal, or pip install "
                "ttnn, or set -DTT_METAL_LIB_DIR.")
        endif()
        message(FATAL_ERROR
            "USE_TENSTORRENT=ON but no usable Tenstorrent installation was found. "
            "See the diagnostic above, or configure without -DUSE_TENSTORRENT=ON.")
    endif()
endif()
