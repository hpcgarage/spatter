## Spatter Benchmark
This is a microbenchmark for timing Scatter/Gather kernels on CPUs and GPUs. 

## Purpose 
For some time now, memory has been the bottleneck in modern computers. As CPUs grow more memory hungry due to increased clock speeds, an increased number of cores, and larger vector units, memory bandwidth and latency continue to stagnate.  While increasingly complex cache hierarchies have helped ease this problem, they are best suited for regular memory accesses with large amounts of locality. However, there are many programs which do not display regular memory patterns and do not reuse data much, and thus do not benefit from such hierarchies. Irregular programs, which include many sparse matrix and graph algorithms, drive us to search  for new approaches to better utilize what little memory bandwidth is available. 

With this benchmark, we aim to characterize the performance of memory systems in a novel way. We want to be able to make comparisons across architectures about how well data can be rearranged, and we want to be able to use benchmark results to predict the runtimes of sparse algorithms on these various architectures. We will use these results to predict the impact of new memory access primitives. 

### Kernels
Spatter supports the following primitives.

Scatter:
    `A[j[:]] = B[:]`

Gather:
    `A[:] = B[i[:]]`

SG:
    `A[j[:]] = B[i[:]]`

### Building
CMake is required to build Spatter

To build with CMake from the main source directory:
```
./configure/configure_ocl
cd build_ocl
make
```
or use one of the other configure scripts to compile with different backends. 

### Quick Start

The only required argument to spatter is the amount of data to move. It will guess all other arguments such as kernel and device. 

```
./spatter -l 2048
```

### Run Your Own Platform Comparison

You can quickly compare one of your platforms to some of the GPUs we have tested on. We will add much more flexibility to this in the future, but for now, we will assume you are using CUDA. 

You must have R installed to generate the plot. 

Steps:

1. You will need the bandwidth of your GPU. If you don't know it, you can go to `tests/run_babel_stream.sh` and run it. The results will be in `tests/BabelStream-3.3/babelstream_DEVICENAME_cuda.txt`. Note the max copy bandwidth.

2. Go to your build folder and run `sparsity_test.sh`. This will take a while. (But it will be optimized soon!) 

3. Go to the `quickstart` directory (sibling of your build directory) and run `./gather_comparison.sh ../build/GATHER_FILE.sh sg_sparse_roofline_cuda_user_GATHER.ssv <BANDWIDTH>`, where BANDWIDTH is the bandwidth from step 1. 

4. This will produce `gather_comparison.eps` in the `quickstart` directory. Your device will be called "USER", and will be colored orange.

![Alt text](resources/gather_comparison.eps?raw=true "Title")

### Arguments
Spatter has a large number of arguments. To start with, you should focus on -k (the kernel), -l (the length of the index arrays), -v (the work per thread) and -z (the CUDA/OpenCL block size).
```
./spatter <arguments>
    -b, --backend=<backend>
        Specify backend: OpenCL or OpenMP
    -p, --cl-platform=<platform>
        Specify platform if using OpenCL (case-insensitve, fuzzy matching)
    -d, --cl-device=<device>
        Specify device if using OpenCL (case-insensitve, fuzzy matching)
    --interactive
        Tell spatter you want to pick the platform and device interactively
    -f, --kernel-file=<file>
        Specify the location of a kernel file
    -k, --kernel-name=<name>
        Specify the name of the kernel (scatter, gather, or sg)  you want to run
    -v, --vector-len 
        Specifies the work per thread (poorly named, sorry)
    -l, --generic-len
        The number of elements to move. Automacially sets source-len, target-len, and index-len based on the kernel
    -W, --workers 
        The number of OMP threads to use
    -w, --wrap
        More info coming soon
    -s, --sparsity
        Sparsity of soruce or target buffers
    -z
        GPU, OpenCL block size
    -q
        Supress warnings
    -nph, --no-print-header
        Don't print the header on the output
    --validate
        Check the output of the kernel against naive CPU output
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
        Not yet implemented
```
