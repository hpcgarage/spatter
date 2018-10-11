#!/bin/bash
#This script runs the R plotting script on the commandline and places output with the data

set -x

#Backend language
BACKEND=$1
#System descriptive name
SYS=$2
#Is the device a CPU, GPU, or KNL?
DEV=$3

RESULTDIR=../results/v0.2

SYSDESC=${BACKEND}/$DEV/${SYS}

OPERATIONS="scatter gather sg"

for OP in $OPERATIONS; 
do
#Make uppercase
OPUPPER=${OP^^}


FULLRESULT=${RESULTDIR}/${OP}/${SYSDESC}

#If this directory doesn't exist, create it. We assume the top-level directory exists
if [ ! -d "$FULLRESULT" ]; then
	mkdir ${RESULTDIR}/${OP}/${BACKEND}
	mkdir ${RESULTDIR}/${OP}/${BACKEND}/${DEV}
	mkdir ${RESULTDIR}/${OP}/${BACKEND}/${DEV}/${SYS}
fi

echo $FULLRESULT

mv sg_sparse_roofline_${BACKEND}_${SYS}_${OPUPPER}.ssv  ${FULLRESULT}/.
done
