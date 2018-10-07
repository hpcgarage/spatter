# Authors: Jeffrey Young, Patrick Lavin
# Last Modified: September 24, 2018
# Create the data for a roofline mode with various sparsities

#Print out each command for debugging purposes
#set -x

SCRIPT='./'`basename "$0"`
USAGE='\nUsage:\n '$SCRIPT' <device>\n'

# Check arguments to script.
if [ $# -lt 1 ]; then
    echo -ne $USAGE
    exit
fi

#User specifies backend
BACKEND=cuda
DEVICE=$1

echo backend=${BACKEND} device=${DEVICE}, platform=${PLATFORM}


#Specify the sparsities you want to test for 
SPARSITY="1 2 4 8 16 32 64 128"
#Specify the backend - openmp, cuda, opencl

BLOCK="1 2 4 8 16"

SHMEM=`seq 0 15 | tac`

SCRIPTNAME=sg_volkov

O_S=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SCATTER.ssv
O_G=${SCRIPTNAME}_${BACKEND}_${DEVICE}_GATHER.ssv
O_SG=${SCRIPTNAME}_${BACKEND}_${DEVICE}_SG.ssv

R=$RANDOM

NVPLOG=log$R.txt
TMP1=tmp1$R
TMP2=tmp2$R
NVPOPT='--aggregate-mode off --log-file '$NVPLOG' --csv --metrics achieved_occupancy'
WHICH=`seq -s',' 15 3 42`

#Specify a large region to be used for the "sparse space
LEN=$((2**20))

for S in $SPARSITY;
do
    for B in $BLOCK;
    do
        for logM in $SHMEM;
        do
            M=$((2**logM))
            ./sgbench -l$LEN -s$S -k scatter -nph -q -v16 -z$B -m$M > $TMP1
            nvprof $NVPOPT ./sgbench -l$LEN -s$S -k scatter -nph -q -v16 -z$B -m$M > /dev/null
            cat $NVPLOG | cut -sd',' -f$WHICH | sed '/^$/d' | tr ',' '\n' > $TMP2
            paste -d' ' $TMP1 $TMP2 >> $O_S

            ./sgbench -l$LEN -s$S -k gather  -nph -q -v16 -z$B -m$M > $TMP1
            nvprof $NVPOPT ./sgbench -l$LEN -s$S -k gather  -nph -q -v16 -z$B -m$M > /dev/null
            cat $NVPLOG | cut -sd',' -f$WHICH | sed '/^$/d' | tr ',' '\n' > $TMP2
            paste -d' ' $TMP1 $TMP2 >> $O_G

            ./sgbench -l$LEN -s$S -k sg      -nph -q -v16 -z$B -m$M > $TMP1
            nvprof $NVPOPT ./sgbench -l$LEN -s$S -k sg      -nph -q -v16 -z$B -m$M > /dev/null
            cat $NVPLOG | cut -sd',' -f$WHICH | sed '/^$/d' | tr ',' '\n' > $TMP2
            paste -d' ' $TMP1 $TMP2 >> $O_SG
        done
    done
done

rm $NVPLOG $TMP1 $TMP2 

# cat log.txt | cut -sd',' -m$WHICH | sed '/^$/d' | tr ',' '\n'

# nvprof --log-file log.txt --csv --metrics achieved_occupancy ./sgbench -l $((2**20)) -m 1 -q -nph -z 1 -v16 -s 2 -q -nph  >> testout.txt
