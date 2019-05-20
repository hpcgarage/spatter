#!/bin/bash

#PBS -N spatter
#PBS -l walltime=06:10:00

#PBS -l place=scatter,select=1:nodetype=mom-x86_64+1:nodetype=SK48

HOST=`aprun hostname | grep nid`
NODE=kay_${HOST: -3}
ARCH=`pbsnodes $NODE | grep nodetype | rev | cut -f1 -d' ' | rev`
MEM=`pbsnodes $NODE | grep availmem | rev | cut -f1 -d' ' | rev`
CLOCK=`pbsnodes $NODE | grep clockmhz | rev | cut -f1 -d' ' | rev`
AVCPU=`pbsnodes $NODE | grep available.ncpus | rev | cut -f1 -d' ' | rev`
ASCPU=`pbsnodes $NODE | grep assigned.ncpus | rev | cut -f1 -d' ' | rev`

HOST=$HOST NODE=$NODE ARCH=$ARCH MEM=$MEM CLOCK=$CLOCK ASCPU=$ASCPU AVCPU=$AVCPU aprun /home/users/plavin/spatter/build_omp_cce/ustride_tests.sh
