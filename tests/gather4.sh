CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=opencl --source-len=256 --target-len=256 --index-len=256 --kernel-file=kernels/gather4.cl --kernel-name=gather --cl-platform=nvidia --cl-device=titan --validate --block-len=1 --vector-len=4