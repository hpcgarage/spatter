#!/bin/bash

len=10000000
./spatter -p "4,4,4,4" -l$len
./spatter -p UNIFORM:8:4 -l$len
./spatter -p UNIFORM:7:7 -d -l$len
./spatter -p MS1:8:3,5:2,2 -l$len
./spatter -p MS1:8:4,7:8 -l$len
./spatter -p MS1:8:4,7 -l$len

