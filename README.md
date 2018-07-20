## Scatter Gather Benchmark {#mainpage}
This is a microbenchmark for timing a Scatter/Gather kernel on CPU. OpenMP is used for multithreading. 

## Purpose 
For some time now, memory has been the bottleneck in modern computers. As CPUs grow more memory hungry due to increased clock speeds, an increased number of cores, and larger vector units, memory bandwidth and latency continue to stagnate.  While increasingly complex cache hierarchies have helped ease this problem, they are best suited for regular memory accesses with large amounts of locality. However, there are many programs which do not display regular memory patterns and do not reuse data much, and thus do not benefit from such hierarchies. Irregular programs, which include many sparse matrix and graph algorithms, drive us to search  for new approaches to better utilize what little memory bandwidth is available. 

With this benchmark, we aim to characterize the performance of memory systems in a novel way. We want to be able to make comparisons across architectures about how well data can be rearranged, and we want to be able to use benchmark results to predict the runtimes of sparse algorithms on these various architectures. We will use these results to predict the impact of new memory access primitives. 

### Building
Currently we have a basic Makefile and a supported CMake infrastructure. 

To build with Make:
```
make
```

To build with CMake from the main source directory:
```
./configure/configure_ocl
cd build_ocl
make
```

[comment]: # (Building with PAPI support requires  `libpapi` to be installed.)  

### Usage
A minimal run of sgbench should specify the following 
    1. backend
    2. If backend is OpenCL: cl-platform, cl-device, kernel-name, kernel-file
    3. source-len, target-len, index-len

```
./sg_bench <arguments>
    --backend=<backend>
        Specify backend: OpenCL or OpenMP
    --cl-platform=<platform>
        Specify platform if using OpenCL (case-insensitve, fuzzy matching)
    --cl-device=<device>
        Specify device if using OpenCL (case-insensitve, fuzzy matching)
    --interactive
        Tell sgbench you want to pick the platform and device interactively
    --kernel-file=<file>
        Specify the location of a kernel file
    --kernel-name=<name>
        Specify the name of the kernel in <kernel-file> you want to run
    --source-len=<blocks>
        The number of blocks that can be moved (default block size is 1)
    --target-len=<blocks>
        The number of blocks that can be filled
    --index-len=<blocks> 
        The number of blocks that will be moved
    --seed=<seed>
        Optional: Specify random seed
    --runs=<count> 
        Specify how many times to run the benchmark (default 10)
    --loops=<count> 
        Specify how many scatters/gathers will be performed by a single run of the benchmark
    --verbose
        Not yet implemented
```
### Paper
The Overleaf repo for the paper is located [here](https://www.overleaf.com/15470014dvbkpnfjpjzm#/58680363/).
#### Arguments

###References
