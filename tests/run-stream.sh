#!/bin/bash
#Download and run STREAM
#Usage: ./run-stream.sh condesa-SNB openmp
#Usage2: ./run-stream.sh wingtip-P100 opencl

#Global variables - assign the commandline parameters here
STREAM=STREAM
SYSDESC=$1
BACKEND=$2


download_stream(){
#Download Stream git repo
git clone https://github.com/jeffhammond/STREAM.git
}


check_param()
{
#Check to make sure two arguments were passed
if [ -z "$SYSDESC" ]; then 
echo "Please pass an identifier like '<machinename-CPU/CPU>' as the first argument"
	exit 1
fi
}

run_stream(){

#Set the number of iterations to run Stream
N=100

cd STREAM

#Change the default GCC/GFortran
sed -i -e 's/gcc-4.9/gcc/g' Makefile
sed -i -e 's/gfortran-4.9/gfortran/g' Makefile
#Update the flags
sed -i -e 's|CFLAGS = -O2 -fopenmp|CFLAGS = -O2 -fopenmp -DSTREAM_ARRAY_SIZE=126000000 -DNTIMES=100|g' Makefile


#Just build the C executable
EXE=stream_c.exe
make ${EXE}

OUTPUTFILE=stream_${SYSDESC}.txt

echo "Running STREAM with n = $N"
#Run OpenMP version of BabelStream on one socket using local allocation
numactl -N 0 -l ./${EXE} &> ${OUTPUTFILE}

RESULTSDIR=../../results/STREAM/
cp ${OUTPUTFILE} ${RESULTSDIR}

#Go back to base directory
cd ..
}

clean_stream(){
rm -rf STREAM
}

#Execute each function or comment out specific functions
check_param
download_stream
run_stream
#clean_stream
