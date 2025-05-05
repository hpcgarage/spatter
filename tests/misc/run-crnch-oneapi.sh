#!/bin/bash
#SBATCH -Jspatter-ci-oneapi                        # Job name
#SBATCH -N1 --cpus-per-task=4                	 # Number of nodes and CPUs per node required
#SBATCH --mem-per-cpu=4G                         # Memory per core
#SBATCH -t 00:30:00                              # Duration of the job (Ex: 30 mins)
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o /tools/ci-reports/spatter-oneapi-test-%j.out   # Combined output and error messages file
#SBATCH --gres gpu:max_1100:1	  	                 # Request a A100 GPU on any available node
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
cd $GITHUB_WORKSPACE
hostname
#This line allows the GH runner to use the module command on the targeted node
source /tools/misc/.read_profile
#Load OneAPI tools
. /opt/intel/oneapi/2025.1/oneapi-vars.sh
cmake -DBACKEND=oneapi -DCOMPILER=icpx -B build_oneapi_workflow -S .
make -C build_oneapi_workflow
cd build_oneapi_workflow
make test