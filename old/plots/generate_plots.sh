#!/bin/bash
# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: October 11, 2018
#This script runs the R plotting script on the commandline and places output with the data
#Ex: ./generate_plots.sh openmp condesa-snb 18158 cpu

#Print out each command for debugging purposes
set -x

SCRIPT='./'`basename "$0"`
USAGE='\nUsage: \n  '$SCRIPT' <BACKEND=openmp/cuda/opencl> <SYSTEM_DESCRIPTION> <STREAMBW=Stream bandwidth MB/s> <DEVICE=cpu/gpu/knl> \n  Ex: '$SCRIPT' cuda octane-k40 120000 gpu\n\n'

# Check arguments to script.
if [ $# -lt 4 ]; then
    echo -ne $USAGE
    exit
fi

#Backend language
BACKEND=$1
#System descriptive name
SYS=$2
#Stream bandwith
BW=$3
#Is the device a CPU, GPU, or KNL?
DEV=$4

CURRTAG=`git describe --tags --abbrev=0`
#CURRTAG=v0.3
RESULTDIR=../results/${RESULTDIR}

SYSDESC=${BACKEND}/${DEV}/${SYS}

OPERATIONS="scatter gather sg"

for OP in $OPERATIONS; 
do

OPUPPER=${OP^^}
FULLRESULT=${RESULTDIR}/${OP}/${SYSDESC}

FILES="sparse rdm"
for F in $FILES
do
	FILENAME=sg_${F}_roofline_${BACKEND}_${SYS}_${OPUPPER}.ssv

	echo $FULLRESULT
	echo $FILENAME

	Rscript --vanilla roofline.R  ${FULLRESULT}/${FILENAME}  ${BACKEND} ${SYS} 0 ${BW} ${FULLRESULT}/${SYS}_${OP}_${F}_v02.png

	done
done
