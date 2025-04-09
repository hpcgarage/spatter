# Spatter Release Notes

## Spatter 2.0 Release Notes

The 2.0 release of Spatter brings several important changes to the codebase, some of which are summarized in the MEMSYS 2024 paper, by Sheridan, et al., ["A Workflow for the Synthesis of Irregular Memory Access Microbenchmarks"](https://www.memsys.io/wp-content/uploads/ninja-forms/5/sheridan_et_al_workflow_irregular_patterns_paper_32_MEMSYS_2024-1.pdf). Specifically, this release includes these major changes:

- Switch from C to C++ for the codebase, enabling more robus configuration options
- Refactor of the CMake infrastructure
- Different parsing mechanisms and deprecation of older parsing code requiring third-party submodules
- Improvements to the CUDA backend including support for longer patterns and support for GPU throughput testing
- MPI support for weak and strong scaling test scenarios
- Removal of integrated support for PAPI
- Addition of contributors guide, improved GH actions for testing, and templates for PRs and new issues.
- Update to the Spatter wiki to describe [changes in Spatter 2.0 that may affect benchmark performance](https://github.com/hpcgarage/spatter/wiki/Spatter-2.0-Validation) and a [new guide for developing new backends](https://github.com/hpcgarage/spatter/wiki/Adding-New-Backends-to-Spatter).

This release includes commits from Patrick Lavin, Jeff Young, Julio Agustin Vaca Valverde, Jered Dominguez-Trujillo, and Connor Radelja. We are greatly appreciative of the support of contributors from Los Alamos National Labs through [their work with Spatter](https://github.com/lanl/spatter) as well as new work on [generating input patterns with GS Patterns](https://github.com/lanl/gs_patterns/).


## Spatter 1.0 Release Notes

After 6 years of development, we are ready to release Spatter version 1.0! Over the years we have gone through a number of input and output formats, but they have not changed significantly in several years now, so we are confident that they are stable enough for our first major release. 

Building Spatter has been a collaborative effort. This release includes commits from Patrick Lavin, Jeff Young, Julio Agustin Vaca Valverde, Jered Dominguez-Trujillo, James Wood, Vincent Huang, Sudhanshu Agarwal, Jeff Inman, and Jeff Hammond. Thank you to all of you for your effort and to those who submitted issues and advised us on the project as well.

### New Features
Since version 0.6, Spatter has added several major new features: 

 - multi-gather, multi-scatter: the multi- kernels perform two levels of indirection, e.g. multi-gather is dest[i] = src[idx1[idx2[i]] and multi-scatter is analogous. This greatly expands the class of patterns that Spatter can represent. 
 - Binary trace support allows for compressed inputs to be used with Spatter to represent synthetic and application traces.
MPI support allows Spatter to run the same pattern on many ranks and allows for weak scaling studies
 - CMake support has been modernized and configuration is greatly simplified. Bespoke configuration scripts for different compilers/backends have been folded into CMake options.
 - Testing and CI/CD has been greatly improved. The GPU backend is now included in the test suite, and we have expanded the set of automated tests used for the CPU backend as well.
 - Documentation now includes a Getting Started Jupyter notebook to demonstrate how to use the benchmark and how to plot outputs.

### Ongoing Work
We are still implementing a CI solution to automatically test commits to the GPU backend. 

