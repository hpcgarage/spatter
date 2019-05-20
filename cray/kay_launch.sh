#!/bin/bash

SUPPORTED="skx bdw hsw csx tx2"

if [ "$#" -ne 1 ]; then
    echo -e "\nUsage: $0 <arch>\n  where <arch> is one of [$SUPPORTED]\n"
    exit
fi

if [ `[[ " $SUPPORTED " =~ " $1 " ]] && echo '1' || echo '0'` -ne 1 ]; then
    echo "Processor not supported"
    exit
fi

if [ "$1" = "skx" ]; then 
    qsub -l place=scatter,select=1:nodetype=mom-x86_64+1:nodetype=SK48 -v PROC=$1 ./submit.sh
fi

if [ "$1" = "bdw" ]; then 
    qsub -l place=scatter,select=1:nodetype=mom-x86_64+1:nodetype=BW36 -v PROC=$1 ./submit.sh
fi

if [ "$1" = "hsw" ]; then 
    qsub -l place=scatter,select=1:nodetype=mom-x86_64+1:nodetype=HW28 -v PROC=$1 ./submit.sh
fi

if [ "$1" = "csx" ]; then 
    qsub -l place=scatter,select=1:nodetype=mom-x86_64+1:nodetype=CL40 -v PROC=$1 ./submit.sh
fi

if [ "$1" = "tx2" ]; then 
    qsub -l place=scatter,select=1:nodetype=mom-aarch64+1:nodetype=TX26 -v PROC=$1 ./submit.sh
fi

