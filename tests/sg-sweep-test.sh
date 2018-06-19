# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: June 12, 2018
# Do a sweep over multiple iterations and options for the benchmark

#Output file
OUTPUT=$1
#Num workers
WORKERS="1 2 4"
#Block sizes to sweep
BLKSIZE="1 2 4 8 16 32"
#source length
SRCSIZE=`seq 10 15`
#dest length
DST=$((2**10))
#index length
IDX=$((2**10))

#op - accumulate versus standard
OP="COPY ACCUM"

#kernel = scatter/gather/s+g

for B in $BLKSIZE
do

    for S in $SRCSIZE
    do
        SRC=$((2**S))

        for O in $OP
        do
            for W in $WORKERS
            do
                CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=opencl --source-len=$SRC --target-len=$DST --index-len=$IDX --kernel-file=kernels/sg.cl --kernel-name=sg --cl-platform=nvidia --cl-device=titan --runs=10 --block-len=$B -W$W --op=$O

            done
        done

    done

done

#execute the test script with nested for loops
#./test-ocl.sh <all of these parameters
