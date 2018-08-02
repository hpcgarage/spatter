CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=openmp --source-len=8388608 --target-len=4194304 --index-len=4194304  --kernel-name=gather --validate --block-len=1

CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=opencl --source-len=8388608 --target-len=4194304 --index-len=4194304 --kernel-file=kernels/gather1.cl --kernel-name=gather --cl-platform=intel --cl-device=xeon --validate --block-len=20 

CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=opencl --source-len=8388608 --target-len=4194304 --index-len=4194304 --kernel-file=kernels/gather16.cl --kernel-name=gather --cl-platform=intel --cl-device=xeon --validate --block-len=20  --vector-len=16
