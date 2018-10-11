#!/bin/bash
#Download and run STREAM
#Usage: ./run-stream.sh condesa-SNB openmp
#Usage2: ./run-stream.sh wingtip-P100 opencl

#set -x

#Global variables - assign the commandline parameters here
STREAM=STREAM
SYSDESC=$1

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

#Update the flags to set size and iterations
sed -i -e 's|CFLAGS = -O2 -fopenmp|CFLAGS = -O2 -fopenmp -DSTREAM_ARRAY_SIZE=126000000 -DNTIMES=100|g' Makefile
#Tuned for ThunderX2 systems
#sed -i -e 's|CFLAGS = -O2 -fopenmp|CFLAGS = -O2 -fopenmp -mtune=-mcpu=thunderx2t99 -mtune=thunderx2t99 -DSTREAM_ARRAY_SIZE=126000000 -DNTIMES=100|g' Makefile


#Just build the C executable
EXE=stream_c.exe
make ${EXE}

OUTPUTFILE=stream_${SYSDESC}.txt

#NUMACTL can be used to run on just one socket but it may conflict with OpenMP env settings
#NUMACTL=numactl -N 0 -l
NUMACTL=

#OpenMP settings to place STREAM on local threads
export OMP_PLACES=sockets
export OMP_PROC_BIND=master
export OMP_DISPLAY_ENV=VERBOSE

#Run OpenMP version of BabelStream on one socket using local allocation
echo "Running STREAM with n = $N"
${NUMACTL} ./${EXE} &> ${OUTPUTFILE}

RESULTSDIR=../../results/STREAM
cp ${OUTPUTFILE} ${RESULTSDIR}/.

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
clean_stream
