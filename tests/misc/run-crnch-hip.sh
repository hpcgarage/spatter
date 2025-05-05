#!/bin/bash
#SBATCH -Jspatter-ci-hip                         # Job name
#SBATCH -N1 --cpus-per-task=4                	 # Number of nodes and CPUs per node required
#SBATCH --mem-per-cpu=4G                         # Memory per core
#SBATCH -t 00:30:00                              # Duration of the job (Ex: 30 mins)
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o /tools/ci-reports/spatter-hip-test-%j.out   # Combined output and error messages file
#SBATCH --gres gpu:mi210:1	  	                 # Request an MI210 GPU on any available node
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
cd $GITHUB_WORKSPACE
hostname
#This line allows the GH runner to use the module command on the targeted node
source /tools/misc/.read_profile
#HIPCC should be enabled on CRNCH GPU nodes without needing module load
cmake -DBACKEND=hip -DCOMPILER=hipcc -B build_hip_workflow -S .
make -C build_hip_workflow
cd build_hip_workflow
make test