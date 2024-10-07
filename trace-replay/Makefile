all: main
trace-replay-kernel.o: trace-replay-kernel.cpp trace-replay-kernel.h
	g++ -fopenmp -c -O3 -o trace-replay-kernel.o trace-replay-kernel.cpp
main: trace-replay-kernel.o main.cpp
	g++ -fopenmp -O3 -o main trace-replay-kernel.o main.cpp
clean:
	rm -rf main
.PHONY: main clean
