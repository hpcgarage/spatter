# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Create the data for a roofline mode with various sparsities

#Print out each command for debugging purposes
set -x

#User specified identifier to clarify the output
DEVICE_NAME=$1

#Specify the vector length - OpenCL allows for specifying double16 for example
VECTOR="1 2 4 8 16 32 64"
#Specify the sparsities you want to test for 
SPARSITY="1 2 4 8 16 32 64 128"
#Specify the backend - openmp, cuda, opencl
BACKEND=$2

SCRIPTNAME=sg_sparse_roofline

O_S=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_SCATTER.ssv
O_G=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_GATHER.ssv
O_SG=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_SG.ssv

#Specify a large region to be used for the "sparse space
LEN=$((2**20))

for V in $VECTOR;
do

    for S in $SPARSITY;
    do
        if [ "${BACKEND}" == "cuda" ]
        then
            ./sgbench -l $LEN -s $S -k scatter -v $V --nph -q 1>> $O_S
            ./sgbench -l $LEN -s $S -k gather  -v $V --nph -q 1>> $O_G
            ./sgbench -l $LEN -s $S -k sg      -v $V --nph -q 1>> $O_SG
        else
            echo "Only CUDA is supported for this script."
        fi
    done

done
