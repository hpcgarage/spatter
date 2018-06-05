#!/bin/bash
#Allow for switching between Intel or NVIDIA OpenCL libraries
#Last updated: 10/6/2016

USERNAME=`whoami`

#Ubuntu 14.04
#PREVPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#Ubuntu 16.04
PREVPATH=/home/$USERNAME/bin:/home/$USERNAME/.local/bin:/home/$USERNAME/bin:/home/$USERNAME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/snap/bin
PREVLIB=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/lib64:/usr/lib

if [[ -z $1 || $1 -eq 2 ]]
then
        echo "Please enter 1, 3, or 4 (Intel SDK, CUDA, OCLGrind) as an argument"
        return
fi

if [ $1 == "1" ];
then
        #SDK has include files but /etc/OpenCl/vendors/intel64.icd points to 
        #other directory for lib
	INTELPATH=/opt/intel/opencl
        #export PATH=/opt/intel/opencl-sdk/bin:/opt/intel/opencl-sdk/gt_debugger_2016.0/bin:${PREVPATH}
        export LD_LIBRARY_PATH=${INTELPATH}/lib64:${PREVLIB}
        export OCL_LIB=${INTELPATH}/lib64
        #export OCL_INCL=/opt/intel/openlclr4/include
	export OCL_INCL=
elif [ $1 == "2" ];
then
        export PATH=:${PREVPATH}
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
	export OCL_INCL=
elif [ $1 == "3" ];
then
        export PATH=/usr/local/cuda/bin:${PREVPATH}
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${PREVLIB}
        export OCL_LIB=/usr/local/cuda/lib64
        export OCL_INCL=/usr/local/cuda/include
elif [ $1 == "4" ];
then
        export PATH=/usr/oclgrind/bin:${PREVPATH}
        export LD_LIBRARY_PATH=/usr/oclgrind/lib:${PREVLIB}
        export OCL_LIB=/usr/oclgrind/lib
        export OCL_INCL=/usr/oclgrind/include
fi

echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "OCL_LIB: $OCL_LIB"
echo "OCL_INCL: $OCL_INCL"
