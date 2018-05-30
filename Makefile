#TODO: Use Jeff's improved makefile
all:
	gcc *.c kernels/openmp_kernels.c -o sgbench -lOpenCL -fopenmp
