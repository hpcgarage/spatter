#!/bin/bash
#Update of Aaron's script using likwid-pin instead of numactl
rm -f omp.txt
rm -f scalar.txt

#loc=".50"
for s in 1 2 4 8 16 32 64 128;
do
	rm -f tmp1.txt tmp2.txt
        for i in `seq 1 10`;
        do
        	OMP_NUM_THREADS=1 numactl -N 0 -l ./spatter -b openmp -s $s -q -R 100 --no-print-header -l 4000000 | cut -d' ' -f10 >> tmp1.txt
		#Weirdly the SERIAL backend doesn't print out "SERIAL" so cut on the 9th column
                OMP_NUM_THREADS=1 numactl -N 0 -l ./spatter -b serial -s $s -q -R 100 --no-print-header -l 4000000 | cut -d' ' -f9 >> tmp2.txt
		#Use likwid-pin
                #OMP_NUM_THREADS=1 likwid-pin -c N:0 ./spatter -b openmp -s $s -q -R 100 --no-print-header -l 4000000 | cut -d' ' -f10 >> tmp1.txt
                #OMP_NUM_THREADS=1 likwid-pin -c N:0 ./spatter -b serial -s $s -q -R 100 --no-print-header -l 4000000 | cut -d' ' -f9 >> tmp2.txt
        done;
        sort -nr tmp1.txt | head -n1 >> omp.txt
        sort -nr tmp2.txt | head -n1 >> scalar.txt

done;

rm -f tmp1.txt tmp2.txt

echo omp.txt:
cat omp.txt
echo scalar.txt:
cat scalar.txt
