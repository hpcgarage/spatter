# Spatter 1.0 Release Notes

After 6 years of development, we are ready to release Spatter version 1.0! Over the years we have gone through a number of input and output formats, but they have not changed significantly in several years now, so we are confident that they are stable enough for our first major release. 

Building Spatter has been a collaborative effort. This release includes commits from Patrick Lavin, Jeff Young, Julio Agustin Vaca Valverde, Jered Dominguez-Trujillo, James Wood, Vincent Huang, Sudhanshu Agarwal, Jeff Inman, and Jeff Hammond. Thank you to all of you for your effort and to those who submitted issues and advised us on the project as well.

## New Features
Since version 0.6, Spatter has added several major new features: 

 - multi-gather, multi-scatter: the multi- kernels perform two levels of indirection, e.g. multi-gather is dest[i] = src[idx1[idx2[i]] and multi-scatter is analogous. This greatly expands the class of patterns that Spatter can represent. 
 - Binary trace support allows for compressed inputs to be used with Spatter to represent synthetic and application traces.
MPI support allows Spatter to run the same pattern on many ranks and allows for weak scaling studies
 - CMake support has been modernized and configuration is greatly simplified. Bespoke configuration scripts for different compilers/backends have been folded into CMake options.
 - Testing and CI/CD has been greatly improved. The GPU backend is now included in the test suite, and we have expanded the set of automated tests used for the CPU backend as well.
 - Documentation now includes a Getting Started Jupyter notebook to demonstrate how to use the benchmark and how to plot outputs.

## Ongoing Work
We are still implementing a CI solution to automatically test commits to the GPU backend. 

