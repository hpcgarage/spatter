#!/bin/bash
#SBATCH -Jspatter-ci-tenstorrent                 # Job name
#SBATCH -N1 --cpus-per-task=4                    # Number of nodes and CPUs per node required
#SBATCH --mem-per-cpu=4G                         # Memory per core
#SBATCH -t 00:30:00                              # Duration of the job (Ex: 30 mins)
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o /tools/ci-reports/spatter-tenstorrent-test-%j.out  # Combined output and error messages file
#SBATCH --gres r5accel:p150a:2                   # Request both Blackhole cards on the node
#SBATCH -W                                       # Do not exit until the submitted job terminates.

# Copied from run-crnch-cuda.sh. Four Tenstorrent-specific notes:
#
#  1. The GRES is r5accel:p150a, NOT gpu -- asking for gpu on this host gets an
#     L40S, which is a different device in the same chassis.
#  2. Both cards are requested even though the tests use one. With a :1
#     allocation the cgroup blocks /dev/tenstorrent/1, and UMD's enumeration
#     walks every card on the host and fails on the blocked one. TT_VISIBLE_DEVICES
#     below is what actually restricts execution to a single chip.
#  3. TT_METAL_RUNTIME_ROOT must point at the directory containing tt_metal/, or
#     tt-metal aborts with "Root Directory is not set" before opening a device.
#  4. -DUSE_TENSTORRENT=ON is the flag the CMake actually consumes. (run-crnch-cuda.sh
#     passes -DBACKEND=cuda -DCOMPILER=nvcc, but BACKEND and COMPILER are not read
#     anywhere in the CMake on main -- see the note in the PR description.)

cd $GITHUB_WORKSPACE
hostname

# This line allows the GH runner to use the module command on the targeted node
source /tools/misc/.read_profile

# tt-metal host library + Python-side runtime root come from the ttnn install.
source ~/ttnn-env/bin/activate
export TT_METAL_RUNTIME_ROOT=$(python3 -c 'import ttnn, os; print(os.path.dirname(ttnn.__file__))')
export TT_VISIBLE_DEVICES=0

cmake -DUSE_TENSTORRENT=ON -B build_tenstorrent_workflow -S .
make -C build_tenstorrent_workflow
cd build_tenstorrent_workflow
make test
