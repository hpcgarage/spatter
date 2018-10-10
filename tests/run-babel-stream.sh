#!/bin/bash
#Download and run STREAM and BabelStream

BSTREAM=3.3
SYSDESC=$1
BACKEND=$2

download_bs(){
#Download BabelStream stable release
wget --no-check-certificate https://github.com/UoB-HPC/BabelStream/archive/v${BSTREAM}.tar.gz
tar xvzf v${BSTREAM}.tar.gz
}


check_param()
{
#Check to make sure two arguments were passed
if [ -z "$SYSDESC" ] || [ -z "$BACKEND" ]; then 
echo "Please pass an identifier like '<machinename-CPU/CPU>' as the first argument and 'openmp' or 'opencl' as the second argument"
	exit 1
fi
}

run_bs(){
#Check for an identifier for the output

cd BabelStream-${BSTREAM}

if [ ${BACKEND} == "openmp" ]; 
then
	make -f OpenMP.make
	EXE=omp-stream
else
	make -f OpenCL.make
	BACKEND=opencl
	EXE=opencl-stream
fi

OUTPUTFILE=babelstream_${SYSDESC}_${BACKEND}.txt

#Set the number of iterations to run BabelStream
N=100
echo "Running BabelStream with n = $N"
#Run OpenMP version of BabelStream on one socket using local allocation
numactl -N 0 -l ./${EXE} -n $N -s $((2**25)) &> ${OUTPUTFILE}

RESULTSDIR=../../results/babelStream/${BACKEND}
cp ${OUTPUTFILE} ${RESULTSDIR}
}

clean_bs(){
#Go up one level and remove this test dir
cd ..
rm -rf BabelStream-${BSTREAM}
rm v${BSTREAM}.tar.gz 
}

#Execute each function or comment out specific functions
check_param
download_bs
run_bs
clean_bs
