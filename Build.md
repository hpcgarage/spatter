# Supported configurations

## Serial 
### Supported Compilers
* gnu
* cray

# Cuda 
### Supported Compilers and optional arguments
* nvcc
    * `-DCUDA_ARCH=<ARCH>`
        * Default is 7.0
        * For example to set it to 7.0, enter `-DCUDA_ARCH=70`

## Openmp
### Supported Compilers and optional arguments
* gnu
    * `-DUSE_MPI=1`
* cray
    * `-DUSE_PAPI=1`
    * `-DUSE_SVE=1`
* clang
* armclang (wombat)
* xl
* intel
    * `-DINTEL_PLATFORM=<PLATFORM>`
        * skylake
        * avx_crossplatform
        * non_avx
    * `-DUSE_MPI=1`
    * `-DUSE_PAPI=1`

## Tenstorrent
Blackhole and other Tenstorrent accelerators, via tt-metal. Enable with
`-DUSE_TENSTORRENT=ON`.

Device kernels are compiled at run time by tt-metal's vendored sfpi toolchain,
so no extra device compiler is needed at build time. The host side needs a C++20
compiler and `libtt_metal`.

### Optional arguments
* `-DTT_METAL_LIB_DIR=<DIR>` - directory holding `libtt_metal.so`
* `-DTT_METAL_INCLUDE_DIRS=<DIR;DIR;...>` - include roots for the host headers

### Runtime environment
* `TT_METAL_RUNTIME_ROOT` must point at the directory containing `tt_metal/`,
  otherwise tt-metal aborts with "Root Directory is not set" before it opens a
  device.
* `TT_VISIBLE_DEVICES` selects which chip(s) to use on a multi-card host.

### If you installed Tenstorrent support with a pip `ttnn` wheel
The wheel ships `libtt_metal.so` and the device-kernel headers, but **not** the
host API headers, so CMake will find the library and then report the headers
missing. They can be assembled without root and without building tt-metal:

```
git clone --depth 1 --branch <version> --filter=blob:none --sparse \
  https://github.com/tenstorrent/tt-metal ~/ttmetal-src
cd ~/ttmetal-src && git sparse-checkout set tt_metal tt_stl
git submodule update --init --depth 1 tt_metal/third_party/umd
```

That covers `tt-metalium`, `tt_stl`, `hostdevcommon` and `umd`. The remaining
four are header-only and are fetched by tt-metal's build rather than vendored,
so clone them at the versions pinned in `~/ttmetal-src/third_party/CMakeLists.txt`:
fmt, nlohmann/json, tt-logger and spdlog, plus enchantum. Pass every include
root via `-DTT_METAL_INCLUDE_DIRS`.

Do **not** run tt-metal's own CMake configure just to obtain these: it pulls
system packages (boost, capnproto, protobuf) that require root, whereas the
header-only clones do not.

### Notes
* Blackhole has no FP64 fabric. This does not affect Spatter, because gather and
  scatter perform no arithmetic; each element is moved as an opaque 8-byte
  payload.
* The DRAM allocator aligns pages to 64 bytes, so a buffer with one 8-byte
  element per page occupies 8x its logical size on device. Size `-l` accordingly.
* `gather` and `scatter` are implemented. The `multi_*` and atomic variants are
  not yet, and report so at run time.

