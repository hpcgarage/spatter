#!/bin/bash
# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: October 11, 2018
#This script organizes data into the correct results directories

#Print out each command for debugging purposes
#set -x

SCRIPT='./'`basename "$0"`
USAGE='\nUsage: \n  '$SCRIPT' <BACKEND=openmp/cuda/opencl> <SYSTEM_DESCRIPTION> <DEVICE=cpu/gpu/knl> \n  Ex: '$SCRIPT' cuda octane-k40 gpu\n\n'

# Check arguments to script.
if [ $# -lt 3 ]; then
    echo -ne $USAGE
    exit
fi

#User specifies backend

#Backend language
BACKEND=$1
#System descriptive name
SYS=$2
#Is the device a CPU, GPU, or KNL?
DEV=$3
#Place results based on the current Github tag
#This assumes that you are working from a tagged branch and that you have git installed and loaded
#You can also manually set this value
CURRTAG=`git describe --tags --abbrev=0`
#CURRTAG=`v0.3`

RESULTDIR=../results/${CURRTAG}

SYSDESC=${BACKEND}/$DEV/${SYS}

OPERATIONS="scatter gather sg"

for OP in $OPERATIONS; 
do
#Make uppercase
OPUPPER=${OP^^}


FULLRESULT=${RESULTDIR}/${OP}/${SYSDESC}

#If this directory doesn't exist, create it. We assume the top-level directory exists
if [ ! -d "$FULLRESULT" ]; then
	mkdir -p ${RESULTDIR}
	mkdir -p ${RESULTDIR}/${OP}
	mkdir -p ${RESULTDIR}/${OP}/${BACKEND}
	mkdir -p ${RESULTDIR}/${OP}/${BACKEND}/${DEV}
	mkdir -p ${RESULTDIR}/${OP}/${BACKEND}/${DEV}/${SYS}
fi

echo "Moving data to results folder at $FULLRESULT"

echo "mv sg_sparse_roofline_${BACKEND}_${SYS}_${OPUPPER}.ssv  ${FULLRESULT}/."
mv sg_sparse_roofline_${BACKEND}_${SYS}_${OPUPPER}.ssv  ${FULLRESULT}/.
mv sg_rdm_roofline_${BACKEND}_${SYS}_${OPUPPER}.ssv  ${FULLRESULT}/.
done
