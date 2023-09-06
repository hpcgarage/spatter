#!/bin/bash
#SBATCH -Jspatter-ci-cuda                        # Job name
#SBATCH -N1 --ntasks-per-node=4                  # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=4G                         # Memory per core
#SBATCH -t 00:30:00                              # Duration of the job (Ex: 30 mins)
#SBATCH -p rg-gpu                                # Partition Name
#SBATCH -o spatter-cuda-test.out                 # Combined output and error messages file
#SBATCH --nodelist quorra1			             # Specify a specific node
#SBATCH -G 1					                 # Request a GPU on that node
#SBATCH -W                                       # Do not exit until the submitted job terminates.

cd $SLURM_SUBMIT_DIR                            # Change to working directory
echo "Current working directory is $(pwd)"

##Add commands here to build and execute
module load cuda
cd ../../
cmake -DBACKEND=cuda -DCOMPILER=nvcc -B build_cuda_workflow -S .
make -C build_cuda_workflow
cd build_cuda_workflow
make test