# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Do a sweep over multiple iterations and options for the benchmark

#Print out each command for debugging purposes
set -x

#User specified identifier to clarify the output
DEVICE_NAME=$1

BLKSIZE="1 2 4 8 16 32"
#source length
SRCSIZE=`seq 10 15`
#Specify the sparsity pattern used
SPARSITY="1 2 3 4 6 8 12 16 24 32 48 64 96 128"
#Specify the vector length - OpenCL allows for specifying double16 for example
VECTOR="1 2 4 8 16"
#Specify the backend - openmp, cuda, opencl
BACKEND=$2
#The number of iterations to run
NUM_RUNS=10


#Specify a large region to be used for the "sparse space
IDX_LEN=$((2**20))

for S in $SPARSITY
do
   for V in $VECTOR
   do
	#Specify the length of the source space
	SRC_LEN=$IDX_LEN
	#Specify the target length to scatter to or gather into
	DST_LEN=$(($IDX_LEN*S))

	OUTPUT_FILE=sg_${BACKEND}_${DEVICE_NAME}.ssv

	#Special handling for the OpenCL case
	if [ "${BACKEND}" == "opencl" ]
	then
   	    CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=$BACKEND --source-len=$SRC_LEN --target-len=$DST_LEN --index-len=$IDX_LEN -kernel-file=kernels/scatter${V}.cl --kernel-name=scatter --cl-platform=nvidia --cl-device=titan --runs=$NUM_RUNS --validate --vector-len=$V &>> $OUTPUT_FILE
        else
   	    ./sgbench --backend=$BACKEND --source-len=$SRC_LEN --target-len=$DST_LEN --index-len=$IDX_LEN --kernel-name=scatter --runs=$NUM_RUNS --validate --vector-len=$V &>> $OUTPUT_FILE
        fi
   done

done
