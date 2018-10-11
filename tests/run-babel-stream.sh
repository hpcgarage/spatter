#!/bin/bash
#Download and run STREAM and BabelStream
#Usage: ./run-babel-stream.sh condesa-SNB openmp
#Usage2: ./run-babel-stream.sh wingtip-P100 opencl

#Global variables - assign the commandline parameters here
BSTREAM=3.3
SYSDESC=$1
BACKEND=$2


download_bs(){
#Download BabelStream stable release and untar it
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

#NUMACTL can be used to run on just one socket but it may conflict with OpenMP env settings
#NUMACTL=numactl -N 0 -l
NUMACTL=

#OpenMP settings to place STREAM on local threads
export OMP_PLACES=threads
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=VERBOSE

#Run OpenMP version of BabelStream on one socket using local allocation
${NUMACTL} ./${EXE} -n $N -s $((2**25)) &> ${OUTPUTFILE}

RESULTSDIR=../../results/babelStream/${BACKEND}
cp ${OUTPUTFILE} ${RESULTSDIR}
}

clean_bs(){
#Go up one level and remove this test dir and the tarball
cd ..
rm -rf BabelStream-${BSTREAM}
rm v${BSTREAM}.tar.gz 
}

#Execute each function or comment out specific functions
check_param
download_bs
run_bs
clean_bs
