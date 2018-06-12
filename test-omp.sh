CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=openmp --source-len=1000 --target-len=1000 --index-len=1000 --kernel-file=kernels/sg.cl --kernel-name=sg  --runs=3 --block-len=30 --validate
