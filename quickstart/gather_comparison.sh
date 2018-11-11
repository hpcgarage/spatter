#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage ./gather_comparison.sh file device_bw(MB/s)"
    exit
fi

#Input files
file1=../results/v0.2/gather/cuda/p100/sg_sparse_roofline_cuda_p100_GATHER_2.ssv 
file2=../results/v0.2/gather/cuda/titan/sg_sparse_roofline_cuda_titan_GATHER_2.ssv 
file3=../results/v0.2/gather/cuda/k40/sg_sparse_roofline_cuda_k40_GATHER.ssv

#Run script
Rscript --vanilla gather_comparison.R $file1 $file2 $file3 $1 0 gather_comparison.eps $2
