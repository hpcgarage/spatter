#!/bin/bash
#This script runs the R plotting script on the commandline and places output with the data

set -x

#Backend language
BACKEND=$1
#System descriptive name
SYS=$2
#Stream bandwith
BW=$3

RESULTDIR=../results/v0.2
SYSDESC=${BACKEND}/cpu/${SYS}

OPERATIONS="scatter gather sg"

for OP in $OPERATIONS; 
do

#OPUPPER=${OP^^}
FULLRESULT=${RESULTDIR}/${OP}/${SYSDESC}

echo $FULLRESULT

Rscript --vanilla roofline.R  ${FULLRESULT}/*.ssv  ${BACKEND} ${SYS} 0 ${BW} ${FULLRESULT}/${SYS}_${OP}_v02.png
done
