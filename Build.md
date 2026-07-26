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
    * `-DSPATTER_ENABLE_NATIVE_ARCH=<ON|OFF>`
        * Default is ON. Tunes the build for the host CPU (adds `-march=native`,
          or `-mcpu=native` where `-march=native` is unsupported), so the
          gather/scatter kernels are compiled for the host ISA (e.g. AVX2/AVX-512).
        * Set to OFF for portable/reproducible binaries or when cross-compiling.
    * `-DSPATTER_ARCH_FLAGS=<flags>`
        * Pin a specific target instead of native detection,
          e.g. `-DSPATTER_ARCH_FLAGS="-march=sapphirerapids"`. Takes precedence
          over `SPATTER_ENABLE_NATIVE_ARCH`.
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
