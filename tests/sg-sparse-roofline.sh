# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Create the data for a roofline mode with various sparsities

#Print out each command for debugging purposes
#set -x

SCRIPT='./'`basename "$0"`
USAGE='\nUsage: \n  '$SCRIPT' openmmp\n  '$SCRIPT' cuda <device>\n  '$SCRIPT' opencl <device> <platform>\n\n'

# Check arguments to script.
if [ $# -lt 1 ]; then
    echo -ne $USAGE
    exit
fi

#User specifies backend
BACKEND=$1


#User specified identifier to clarify the output
if [ $BACKEND == "cuda" ]; 
then
    if [ $# -lt 2 ]; then
        echo -new $USAGE
        exit
    fi
    DEVICE=$2
fi

if [ $BACKEND == "opencl" ]; 
then
    if [ $# -lt 3 ]; then
        echo -ne $USAGE
        exit
    fi
    DEVICE=$2
    PLATFORM=$3
fi


echo backend=${BACKEND} device=${DEVICE}, platform=${PLATFORM}

#Specify the vector length - OpenCL allows for specifying double16 for example
VECTOR="1 2 4 8 16 32 64"
#Specify the sparsities you want to test for 
SPARSITY="1 2 4 8 16 32 64 128"
#Specify the backend - openmp, cuda, opencl

SCRIPTNAME=sg_sparse_roofline

O_S=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SCATTER.ssv
O_G=${SCRIPTNAME}_${BACKEND}_${DEVICE}_GATHER.ssv
O_SG=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SG.ssv

#Specify a large region to be used for the "sparse space
LEN=$((2**20))

export CL_HELPER_NO_COMPILER_OUTPUT_NAG=1

for V in $VECTOR;
do

    for S in $SPARSITY;
    do
        if [ "${BACKEND}" == "cuda" ]
        then
            ./sgbench -l $LEN -s $S -k scatter -v $V --nph -q 1>> $O_S
            ./sgbench -l $LEN -s $S -k gather  -v $V --nph -q 1>> $O_G
            ./sgbench -l $LEN -s $S -k sg      -v $V --nph -q 1>> $O_SG
        elif [ "${BACKEND}" == "opencl" ]
        then
            ./sgbench -p $PLATFORM -d $DEVICE -l $LEN -s $S -k scatter -v $V --nph -q 1>> $O_S
            ./sgbench -p $PLATFORM -d $DEVICE -l $LEN -s $S -k gather  -v $V --nph -q 1>> $O_G
            ./sgbench -p $PLATFORM -d $DEVICE -l $LEN -s $S -k sg      -v $V --nph -q 1>> $O_SG
        else 
            echo "Only CUDA is supported for this script."
        fi
    done

done
