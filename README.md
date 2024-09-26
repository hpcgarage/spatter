# Spatter  

Spatter is a microbenchmark for timing Gather/Scatter kernels on CPUs and GPUs. View the [source](https://github.com/hpcgarage/spatter), and please submit an issue on Github if you run into any issues. 

![Build Status](https://github.com/hpcgarage/spatter/actions/workflows/build-serial-omp.yml/badge.svg)


## Purpose 
For some time now, memory has been the bottleneck in modern computers. As CPUs grow more memory hungry due to increased clock speeds, an increased number of cores, and larger vector units, memory bandwidth and latency continue to stagnate.  While increasingly complex cache hierarchies have helped ease this problem, they are best suited for regular memory accesses with large amounts of locality. However, there are many programs which do not display regular memory patterns and do have much data reuse, and thus do not benefit from such hierarchies. Irregular programs, which include many sparse matrix and graph algorithms, drive us to search for new approaches to better utilize whatever memory bandwidth is available.

With this benchmark, we aim to characterize the performance of memory systems in a new way. We want to be able to make comparisons across architectures about how well data can be rearranged, and we want to be able to use benchmark results to predict the runtimes of sparse algorithms on these various architectures.

### Kernels
Spatter supports the following primitives:

Scatter:
    `A[j[:]] = B[:]`

Gather:
    `A[:] = B[i[:]]`

Concurrent Gather/Scatter:
    `A[j[:]] = B[i[:]]`

MultiScatter:
    `A[j1[j2[:]]] = B[:]`

MultiGather:
    `A[:] = B[i1[i2[:]]]`
    
![Gather Comparison](.resources/sgexplain2.png?raw=true "Gather Comparison")
    
This diagram depicts a combined Gather/Scatter. Gather performs on the top half of this diagram and Scatter the second half.

## Building
CMake is required to build Spatter. Currently we require CMake 3.25 or newer.

To build with CMake from the main source directory, use the following command structure:
```
cmake -DCMAKE_BUILD_TYPE=<BUILD_TYPE> -DUSE_<OPENMP/CUDA/MPI>=1 -B build_<BACKEND> -S .
cd build_<BACKEND>
make
```
For example, to do a debug build with the serial backend:
```
cmake -DCMAKE_BUILD_TYPE=Debug -B build_serial -S .
cd build_serial
make
```
To do an OpenMP and MPI build:
```
cmake -DUSE_OPENMP=1 -DUSE_MPI=1 -B build_openmp_mpi -S .
```

For CUDA builds, we normally load CUDA 11/12 using NVHPC:
```
cmake -DUSE_CUDA=1 -B build_cuda -S .
```
For a complete list of build options, see [Build.md](Build.md)

## Running Spatter
Spatter is highly configurable, but a basic run is rather simple. You must at least specify a pattern with `-p` and you should specify a length with `-l`. Spatter will print out the time it took to perform the number of gathers you requested with `-l` and it will print out a bandwwidth. As a sanity check, the following run should give you a number close to your STREAM bandwith, although we note that this is a one-sided operation - it only performs gathers (reads).
```
./spatter -pUNIFORM:8:1 -l$((2**24))
```

### Notebook for Getting Started

You can quickly compare one of your platforms to some of the CPUs and GPUs we have tested on.

In the `noteboooks/` directory, open up [GettingStarted.ipynb](notebooks/GettingStarted.ipynb). This notebook will guide you through running the standard testsuites found in `standard-suite/`, and it will plot the data for you.

### Arguments
Spatter has a large number of arguments, broken up into two types. Backend configuration options are specied once for each invocation of Spatter, and benchmark configuration arguments can be supplied in bulk using a `.json` file. These arguments may be specified in any order, but it may be simpler if you list all of your backend arguments first. The only required argument to Spatter is `-p`, a benchmark configuration argument.

#### Backend Configuration
Backend configuration arguments determine which language and device will be used. Spatter can be compiled with support for multiple backends, so it is possible to choose between backends and devices at runtime. Spatter will attempt intelliigently pick a backend for you, so you may not need to worry about these arguments at all! It is only necessary to specifiy which `--backend` you want if you have compiled with support for more than one, and it is only necessary to specify which `--device` you want if there would be ambiguity (for instance, if you have more than one GPU available). If you want to see what Spatter has chosen for you, you can run with `--verbose`.

```
$> ./spatter --help

Usage: ./spatter
-a (--aggregate) Aggregate (default off)
   (--atomic-writes) Enable atomic writes for CUDA backend (default 0/off)
-b (--backend) Backend (default serial)
-c (--compress) Enable compression of pattern indices
-d (--delta) Delta (default 8)
-e (--boundary) Set Boundary (limits max value of pattern using modulo)
-f (--file) Input File
-g (--pattern-gather) Set Inner Gather Pattern (Valid with kernel-name: sg, multigather)
-h (--help) Print Help Message
-j (--pattern-size) Set Pattern Size (truncates pattern to pattern-size)
-k (--kernel) Kernel (default gather)
-l (--count) Set Number of Gathers or Scatters to Perform (default 1024)
-m (--shared-memory) Set Amount of Dummy Shared Memory to Allocate on GPUs
-n (--name) Specify the Configuration Name
-p (--pattern) Set Pattern
-r (--runs) Set Number of Runs (default 10)
-s (--random) Set Random Seed (default random)
-t (--omp-threads) Set Number of Threads (default 1 if !USE_OPENMP or backend != openmp or OMP_MAX_THREADS if USE_OPENMP)
-u (--pattern-scatter) Set Inner Scatter Pattern (Valid with kernel-name: sg, multiscatter)
-v (--verbosity) Set Verbosity Level (default 1)
-w (--wrap) Set Wrap (default 1)
-x (--delta-gather) Delta (default 8)
-y (--delta-scatter) Delta (default 8)
-z (--local-work-size) Set Local Work Size (default 1024)
```      

#### Pattern
Spatter supports a few built-in patterns, such as uniform stride, mostly stride-1, and Laplacian. 

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
Laplacian:
    -pLAPLACIAN:<dimension>:<pseudo_order>:<problem_size>
        dimension: The dimension of the stencil
        pseudo_order: The length of a branch of the stencil
        problem_size: The length of each dimension of the problem
        E.g. LAPLACIAN:1:1:100 -> [0,1,2] // 3-point stencil
             LAPLACIAN:2:1:100 -> [0,99,100,101,200] // 5-point stencil
             LAPLACIAN:2:2:100 -> [0,100,198,199,200,201,202,300,400] // 9-point stencil
             LAPLACIAN:3:1:100 -> [0,9900,9999,10000,10001,10100,20000] // 7-point stencil (3D)

        The default delta is 1 for Laplacian patterns
```

You can also simply specify your own pattern, of any length.
```
Custom:
    -p1,2,4,8,16,32
    -p4,4,4,4,4
```

#### JSON Inputs for Multiple Configurations
You may specify multiple sets of benchmark configuration options to Spatter inside a JSON file and run them using `./spatter -pFILE=<jsonconfig>.json`. Examples can be found in the `json/` directory. The file format is below. String values should be quoted while numeric values should not be. 
```
[
    {"long-option1":numeric, "long-option2":"string", ...},
    {"long-option1":numeric, "long-option2":"string", ...},
    ...
]

```

As an example of running with an example JSON configuration. Note that results are provided on a per-pattern basis and summary results are provided for all patterns. This is useful for summarizing pattern results that represent an application kernel. 
```
./spatter -pFILE=../json/ustride_small.json                                                  

Running Spatter version 0.0
Compiler: Clang ver. 7.1.0
Compiler Location: /sw/wombat/ARM_Compiler/19.2/opt/arm/arm-hpc-compiler-19.2_Generic-AArch64_RHEL-7_aarch64-1/bin/armclang         
Backend: OPENMP
Aggregate Results? YES

Run Configurations
[ {'name':'UNIFORM:8:1:NR', 'kernel':'Gather', 'pattern':[0,1,2,3,4,5,6,7], 'delta':8, 'length':2500, 'agg':10, 'wrap':1, 'threads':112},
  {'name':'UNIFORM:8:2:NR', 'kernel':'Gather', 'pattern':[0,2,4,6,8,10,12,14], 'delta':16, 'length':1250, 'agg':10, 'wrap':1, 'threads':112},
  {'name':'UNIFORM:8:4:NR', 'kernel':'Gather', 'pattern':[0,4,8,12,16,20,24,28], 'delta':32, 'length':625, 'agg':10, 'wrap':1, 'threads':112} ]

config  time(s)      bw(MB/s)
0       0.0008033    199.168
1       0.0007809    102.445
2       0.0007738    51.6945

Min          25%          Med          75%          Max
51.6945      51.6945      102.445      199.168      199.168
H.Mean       H.StdErr
87.9079      26.5821
```

For your convienience, we also provide a python script to help you create configurations quickly. If your json contains arrays, you can pass it into the python script `python/generate_json.py` and it will expand the arrays into multiple configs, each with a single value from the array. Given that you probably don't want your pattern arguments to be expanded like this, they should be specified as python tuples. An example is below. 

```
[
    {"kernel":"Gather", "pattern":(1,2,3,4), "count":[2**i for i in range(3)]}
]
   |
   |
   v
[
    {"kernel":"Gather", "pattern":(1,2,3,4), "count":1},
    {"kernel":"Gather", "pattern":(1,2,3,4), "count":2},
    {"kernel":"Gather", "pattern":(1,2,3,4), "count":4}
]
```

## Publications and Citing Spatter

Please see our paper on [arXiv](https://arxiv.org/abs/1811.03743) for experimental results and more discussion of the tool. If you use Spatter in your work, please cite it from the accepted copy from [MEMSYS 2020](https://dl.acm.org/doi/abs/10.1145/3422575.3422794).

Lavin, P., Young, J., Vuduc, R., Riedy, J., Vose, A. and Ernst, D., Evaluating Gather and Scatter Performance on CPUs and GPUs. In The International Symposium on Memory Systems (pp. 209-222). September 2020.

### Other publications
* [SC19 Student Research Competition Poster](https://sc19.supercomputing.org/proceedings/src_poster/src_poster_pages/spostg136.html) ([PDF](https://github.com/hpcgarage/spatter/wiki/pubs/sc19/plavin_spatter_poster_sc19.pdf))
* [SC19 poster abstract (PDF)](https://github.com/hpcgarage/spatter/wiki/pubs/sc19/plavin_spatter_abstract_sc19.pdf)

### Slides
* [SC19 Georgia Tech Booth Talk (PDF)](https://github.com/hpcgarage/spatter/wiki/pubs/sc19/plavin_spatter_booth_talk_sc19.pdf)

## Supported Platforms 

### Linux and Mac 

#### Dependencies: 

* CMake 3.25+ 
* A supported C++ 17 compiler 
  * GCC 
  * Clang 
* If using CUDA, CUDA 11.0+ 
* If using OpenMP, OpenMP 3.0+
  * Note: Issues have been reported in Mac systems with OpenMP. If you encounter issues finding OpenMP when building on Mac OSX, please try to build and run Spatter in a Linux container. 
