CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=opencl --source-len=16 --target-len=32 --index-len=16 --kernel-file=kernels/scatter2.cl --kernel-name=scatter --cl-platform=nvidia --cl-device=titan --validate --vector-len=2
