USEPAPI=0

CC=gcc
EXE=sgbench
SRC=cl-helper.c mytime.c parse-args.c sgbuf.c kernels/openmp_kernels.c

#These includes and libs are usually specified on the command-line using "set_opencl_env.sh" which selects between OpenCL includes and libraries
ifdef OCL_INCL
  CL_CFLAGS = -I${OCL_INCL}
endif

ifdef OCL_LIB
  CL_LDFLAGS = -L${OCL_LIB}
endif

#Link with OpenCL lib and check to see if PAPI should be used
LIBS=-lOpenCL

ifeq (${USEPAPI}, 1)
	LIBS += -lpapi
endif

all: *.c
	$(CC) -DUSEPAPI=${USEPAPI} -O3 ${CL_CFLAGS} ${CL_LDFLAGS} -o ${EXE} $(SRC) main.c  -fopenmp ${LIBS}
test: *.c
	$(CC) -DUSEPAPI=${USEPAPI} -O3 ${CL_CFLAGS} ${CL_LDFLAGS} -o test $(SRC) test.c    -fopenmp ${LIBS}

debug: *.c
	$(CC) -g -std=gnu99 -o ${EXE} *.c kernels/openmp_kernels.c

clean:
	rm -rf ${EXE}

