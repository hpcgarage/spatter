#!/bin/bash

if [ $# -ne 2 ];
then
    echo "Usage: $0 <box> <arch>"
    exit
fi

box=$1
arch=$2
dir=vector_scalar/${box}_${arch}

if [ $arch = "tx2" ];
then
    . ~/.arm_cmake
fi

mkdir -p $dir

echo "Files will be written to $dir/"

if [ $arch = "tx2" ];
then
    cp configure/configure_omp_cce_arm configure_omp
    cp configure/configure_serial_cce_arm configure_serial
else
    cp configure/configure_omp_cce configure_omp
    cp configure/configure_serial_cce configure_serial
fi

./configure_omp
./configure_serial

cd build_omp_cce
make
cd ..

cd build_serial_cce
make
cd ..

cp build_omp_cce/spatter $dir/spatter_v
cp build_serial_cce/spatter $dir/spatter_s

cd $dir

OMP_NUM_THREADS=1 ./spatter_s -pFILE=$HOME/spatter_data/json/ustride_simple.json > scalar_$arch.txt
OMP_NUM_THREADS=1 ./spatter_v -pFILE=$HOME/spatter_data/json/ustride_simple.json > vector_$arch.txt

rm -f spatter_v
rm -f spatter_s
