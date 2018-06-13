# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: June 12, 2018
# Do a sweep over multiple iterations and options for the benchmark

#Output file
OUTPUT=$1

#Block sizes to sweep
BLKSIZE (1 2 4 8 16 32)
#source length

#dest length

#index length

#op - accumulate versus standard
OP

#kernel = scatter/gather/s+g

for BLKSIZE
do

    for SRCSIZE
    do

        for KERNEL
        do

            for OPTYPE
            do

            done



        done

    done

done

#execute the test script with nested for loops
./test-ocl.sh <all of these parameters>
