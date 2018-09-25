# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Create the data for a roofline mode with various sparsities

#Print out each command for debugging purposes
# set -x

#User specified identifier to clarify the output
DEVICE_NAME=$1

#Specify the vector length - OpenCL allows for specifying double16 for example
VECTOR="1 2 4 8 16 32 64"
#Specify the backend - openmp, cuda, opencl
BACKEND=$2

SCRIPTNAME=sg_roofline

O_S=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_SCATTER.ssv
O_G=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_GATHER.ssv
O_SG=${SCRIPTNAME}_${BACKEND}_${DEVICE_NAME}_SG.ssv

#Specify a large region to be used for the "sparse space
LEN=$((2**20))

for V in $VECTOR;
do

    if [ "${BACKEND}" == "cuda" ]
    then
        ./sgbench --backend=cuda --l $LEN -k scatter -v $V --nph -q &>> $O_S
        ./sgbench --backend=cuda --l $LEN -k gather  -v $V --nph -q &>> $O_G
        ./sgbench --backend=cuda --l $LEN -k sg      -v $V --nph -q &>> $O_SG
    else
        echo "Only CUDA is supported for this script."
    fi

done
