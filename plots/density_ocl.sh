#!/bin/bash

#Input files
base=../results/v0.2/gather/opencl
file1=$base/p100/sg_sparse_roofline_opencl_p100_GATHER_2.ssv
file2=$base/titan/sg_sparse_roofline_opencl_titan_GATHER_2.ssv
file3=$base/k40/sg_sparse_roofline_opencl_k40_GATHER_3.ssv
file4=$base/broadwell/sg_sparse_roofline_opencl_wingtip-bdw_GATHER_2.ssv
file5=$base/sandy/sg_sparse_roofline_opencl_condesa-snb_GATHER_3.ssv
#Run script
Rscript --vanilla density.R $file1 $file2 $file3 $file4 $file5 $file6 0 density_opencl.eps

