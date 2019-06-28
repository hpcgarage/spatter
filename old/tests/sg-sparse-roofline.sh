# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Create the data for a roofline mode with various sparsities

#Print out each command for debugging purposes
#set -x

SCRIPT='./'`basename "$0"`
USAGE='\nUsage: \n  '$SCRIPT' openmmp <device> \n  '$SCRIPT' cuda <device>\n  '$SCRIPT' opencl <device> <platform>\n\n'

# Check arguments to script.
if [ $# -lt 1 ]; then
    echo -ne $USAGE
    exit
fi

#User specifies backend
BACKEND=$1

#Exe file
EXE=spatter


#User specified identifier to clarify the output
if [ $BACKEND == "cuda" ] || [ $BACKEND == "openmp" ] ; 
then
    if [ $# -lt 2 ]; then
        echo -ne $USAGE
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
#Note that this number of threads is specific to TX2, Cavium
#NUMTHREADS="1 16 32 64"

BLOCK="1 2 4 8 16"

SCRIPTNAME=sg_sparse_roofline

O_S=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SCATTER.ssv
O_G=${SCRIPTNAME}_${BACKEND}_${DEVICE}_GATHER.ssv
O_SG=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SG.ssv

#Use numactl to allocate memory locally and only use one socket
#NUMACTL="numactl -N 0 -l"
#NUMACTL=

#Specify a large region to be used for the "sparse space". We recommend to use at 
#least 32 MB (2^25) for GPUs and CPUs to mitigate L3 caching or prefetching effects 
LEN=$((2**25))

export CL_HELPER_NO_COMPILER_OUTPUT_NAG=1

for V in $VECTOR;
do

    for S in $SPARSITY;
    do
        if [ "${BACKEND}" == "cuda" ]
        then
            for B in $BLOCK;
            do
                ./${EXE} -l $LEN -d $DEVICE -s $S -k scatter -v $V --nph -q 1 -z $B>> $O_S
                ./${EXE} -l $LEN -d $DEVICE -s $S -k gather  -v $V --nph -q 1 -z $B>> $O_G
                ./${EXE} -l $LEN -d $DEVICE -s $S -k sg      -v $V --nph -q 1 -z $B>> $O_SG
            done
        elif [ "${BACKEND}" == "opencl" ]
        then
            for B in $BLOCK;
            do
                ./${EXE} -p $PLATFORM -d $DEVICE -l $LEN -s $S -k scatter -v $V --nph -q 1 -z $B >> $O_S
                ./${EXE} -p $PLATFORM -d $DEVICE -l $LEN -s $S -k gather  -v $V --nph -q 1 -z $B >> $O_G
                ./${EXE} -p $PLATFORM -d $DEVICE -l $LEN -s $S -k sg      -v $V --nph -q 1 -z $B >> $O_SG
            done
        elif [ "${BACKEND}" == "openmp" ]
        then
	    #for N in $NUMTHREADS;
	    #do
               $NUMACTL ./${EXE} -l $LEN -s $S -k scatter -v $V --nph -q 1>> $O_S
               $NUMACTL ./${EXE} -l $LEN -s $S -k gather  -v $V --nph -q 1>> $O_G
               $NUMACTL ./${EXE} -l $LEN -s $S -k sg      -v $V --nph -q 1>> $O_SG
	    #done
        else 
            echo "Unknown backend" 
        fi
    done

done
