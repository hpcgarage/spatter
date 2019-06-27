# Spatter 
This is a microbenchmark for timing Gather/Scatter kernels on CPUs and GPUs. View the [source](https://github.com/hpcgarage/spatter) and read more abut Spatter in our recently submitted [paper](https://arxiv.org/abs/1811.03743). Please submit an issue on Github if you run into any issues.

## Purpose 
For some time now, memory has been the bottleneck in modern computers. As CPUs grow more memory hungry due to increased clock speeds, an increased number of cores, and larger vector units, memory bandwidth and latency continue to stagnate.  While increasingly complex cache hierarchies have helped ease this problem, they are best suited for regular memory accesses with large amounts of locality. However, there are many programs which do not display regular memory patterns and do not reuse data much, and thus do not benefit from such hierarchies. Irregular programs, which include many sparse matrix and graph algorithms, drive us to search  for new approaches to better utilize what little memory bandwidth is available. 

With this benchmark, we aim to characterize the performance of memory systems in a novel way. We want to be able to make comparisons across architectures about how well data can be rearranged, and we want to be able to use benchmark results to predict the runtimes of sparse algorithms on these various architectures. We will use these results to predict the impact of new memory access primitives. 

### Kernels
Spatter supports the following primitives:

Scatter:
    `A[j[:]] = B[:]`

Gather:
    `A[:] = B[i[:]]`

S+G:
    `A[j[:]] = B[i[:]]`
    
![Gather Comparison](.resources/sgexplain2.png?raw=true "Gather Comparison")
    
This diagram depicts the full Scatter+Gather. Gather performs on the top half of this diagram and Scatter the second half.



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

The only required argument to spatter is the amount of data to move. It will guess all other arguments such as kernel and device. However, this produces data for a single sparsity (default is 1) and doesn't do any tuning. To obtain more useful output, continue on to the next section.

```
./spatter -l 2048
```

### Run Your Own Platform Comparison

You can quickly compare one of your platforms to some of the GPUs we have tested on. We will add much more flexibility to this in the future, but for now, we will assume you are using CUDA. 

You must have R installed to generate the plot. 

Steps:

1. You will need the bandwidth of your GPU. If you don't know it, you can go to `tests/run_babel_stream.sh` and run it. The results will be in `tests/BabelStream-3.3/babelstream_DEVICENAME_cuda.txt`. Note the max copy bandwidth.

2. Go to your build folder (`build_cuda`) and run `sparsity_test.sh`. This will take a while. (But it will be optimized soon!) 

3. Go to the `quickstart` directory (sibling of your build directory) and run `./gather_comparison.sh ../build_cuda/sg_sparse_roofline_cuda_user_GATHER.ssv BANDWIDTH`, where `BANDWIDTH` is the bandwidth from step 1. 

4. This will produce `gather_comparison.eps` in the `quickstart` directory. Your device will be called "USER", and will be colored orange.

![Gather Comparison](.resources/gather_comparison_transparant.png?raw=true "Gather Comparison")

### Arguments
Spatter has a large number of arguments, broken up into two types. Backend configuration options are specied once for each invocation of Spatter, and benchmark configuration arguments can be supplied in bulk using a `.json` file. These arguments may be specified in any order, but it may be simpler if you list all of your backend arguments first. The only reuired argument to Spatter is `-p`, a benchmark configuration argument.

#### Backend Configuration
Backend configuration arguments determine which language and device will be used. Spatter can be compiled with support for multiple backends, so it is possible to choose between backends and devices at runtime. Spatter will attempt intelliigently pick a backend for you, so you may not need to worry about these arguments at all! It is only necessary to specifiy which `--backend` you want if you have compiled with support for more than one, and it is only necessary to specify which `--device` you want if there would be ambiguity (for instance, if you have more than one GPU available). If you want to see what Spatter has chosen for you, you can run with `--verbose`.

```
./spatter <arguments>
    -b, --backend=<backend>
        Specify backend: OpenCL, OpenMP, CUDA, or Serial
    --cl-platform=<platform>
        Specify platform if using OpenCL (case-insensitve, fuzzy matching)
    --cl-device=<device>
        Specify device if using OpenCL (case-insensitve, fuzzy matching)
    --interactive
        Pick the platform and device interactively
    -f, --kernel-file=<file>
        Specify the location of an OpenCL kernel file
    -q, --no-print-header
        Do not print header info. (May be repeated up to 3 times.)
    --verbose
        Print info about default arguments that you have not overridden
    --aggregate=<0,1>
        Report a min time over all runs of a given configurations [Default 1] (Do not use with PAPI) 
```
        
        

#### Benchmark Configuration
The second set of arguments are benchmark  configuration arguments, and these define how the benchmark is run, including the pattern used and the amount of data that is moved. These arguments are special because you can supply multiple sets of benchmark configurations to spatter so that many runs can be performed at once. This way, memory is allocated only once which greatly reduces the amount of time needed to collect a large amount of data.

```
./spatter <arguments>
    -p, --pattern=<Built-in pattern>
    -p, --pattern=FILE=<config file>
        See the section on Patterns. 
    -k, --kernel-name=<kernel>
        Specify the kernel you want to run [Default: Gather]
    -d, --delta=<delta[,delta,...]>
        Specify one or more deltas [Default: 8]
    -l, --count=<N>
        Number of Gathers or Scatters to do
    -w, --wrap=<N>
        Number of independent slots in the "small" buffer (Source buffer if Scatter, Target buffer if Gather) [Default: 1]
    -R, --runs=<N>
        Number of times to repeat execution of the kernel. [Default: 10]
    -t, --opm-thread=<N>
        Number of OpenMP threads [Default: OMP_MAX_THREADS]
    -z, --local-work-size=<N>
        Number of Gathers or Scatters performed by each thread on a GPU
    -s, --shared-memory=<N>
        Amount of dummy shared memory to allocate on GPUs (used for occupancy control)
    -n, --name=<NAME>
        Specify and name used to identify this configuration in the output
    
```

### Pattern
Spatter supports two built-in pattners, uniform stride and mostly stride-1. 

```
Uniform:
    -pUNIFORM:<length>:<gap>
        Length is the length of the pattern, and gap is the size of each jump. 
        E.g. UNIFORM:8:4 -> [0,4,8,12,16,20,24,28]
Mostly Stride-1
    -pMS1:<length>:<gap_locations>:<gap(s)>
        Length is the length of the pattern, gap_locations are the places within the pattern
        with a non-1 gap, and gap are the size of those gaps.  If more than one gap_location 
        is specified, but only a single gap, the gap will be reused. 
        E.g. MS1:8:4:32 -> [0,1,2,3,35,36,37,38]
             MS1:8:2,3:20 -> [0,1,21,41,42,43,44,45]
             MS1:8:2,3:20,22 -> [0,1,21,43,44,45,46,47]

```

You can also simply specify your own pattern, of any length.
```
Custom:
    -p1,2,4,8,16,32
    -p4,4,4,4,4
```

