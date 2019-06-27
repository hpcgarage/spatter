#!/bin/bash

len=10000000
./spatter -p "4,4,4,4" -l$len
./spatter -p UNIFORM:8:4 -l$len
./spatter -p UNIFORM:7:7 -d -l$len
./spatter -p MS1:8:3,5:2,2 -l$len
./spatter -p MS1:8:4,7:8 -l$len
./spatter -p MS1:8:4,7 -l$len
./spatter -pUNIFORM:16:12 -d195,191,187,195,191,289 -w1 -l29489
./spatter -pUNIFORM:13:12 -d4 -w8 -l29489

OMP_NUM_THREADS=8 ./spatter -pUNIFORM:13:12 -d4 -w8 -l29489 -t2 -R3
OMP_NUM_THREADS=8 ./spatter -pUNIFORM:13:12 -d4 -w8 -l29489 -t3 -R3
OMP_NUM_THREADS=8 ./spatter -pUNIFORM:13:12 -d4 -w8 -l29489     -R3
