# Supported configurations

## Serial 
### Supported Compilers
* gnu
* cray

# Cuda 
### Supported Compilers
* nvcc

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