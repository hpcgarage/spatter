# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Run one test where the user can manually pass all the parameters multiple iterations and options for the benchmark

#Print out each command for debugging purposes
set -x

#User specified identifier to clarify the output
DEVICE_NAME=$1
#Specify the sparsity pattern used
SPARSITY=2
#SPARSITY=$2
#Specify the vector length - OpenCL allows for specifying double16 for example
VECTOR=4
#VECTOR=$3
#Specify the backend - openmp, cuda, opencl
BACKEND=cuda
#BACKEND=$4
#The number of iterations to run
NUM_RUNS=10
#NUM_RUNS=$5


#Specify a large region to be used for the "sparse space
IDX_LEN=$((2**20))
#Specify the length of the source space
SRC_LEN=$IDX_LEN
#Specify the target length to scatter to or gather into
DST_LEN=$(($IDX_LEN*SPARSITY))

#Specify a single output file
OUTPUT_FILE=sg_${BACKEND}_${DEVICE_NAME}.ssv

#Special handling for the OpenCL case
if [ "${BACKEND}" == "opencl" ]
then
   	    CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=$BACKEND --source-len=$SRC_LEN --target-len=$DST_LEN --index-len=$IDX_LEN -kernel-file=kernels/scatter${VECTOR}.cl --kernel-name=scatter --cl-platform=nvidia --cl-device=titan --runs=$NUM_RUNS --validate --vector-len=$VECTOR &>> $OUTPUT_FILE
else
   	    ./sgbench --backend=$BACKEND --source-len=$SRC_LEN --target-len=$DST_LEN --index-len=$IDX_LEN --kernel-name=scatter --runs=$NUM_RUNS --validate --vector-len=$VECTOR &>> $OUTPUT_FILE
fi
