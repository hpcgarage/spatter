#!/bin/bash

source /home/users/plavin/python/env/bin/activate

len=$((2**20))
echo $len

#outfile=out_skx.txt
SPATTER_BUILD=/home/users/plavin/spatter/build_omp_cce


# Real Parameters
v=32
reuse=16
run=5
loop=10
outfile=$SPATTER_BUILD/out_skx_full.txt


# Testing Parameters
v=3
reuse=4
run=2
loop=5
outfile=$SPATTER_BUILD/out_skx_test.txt

rm -f $outfile
echo "##################################" >> $outfile
echo "### UNIFORM STRIDE NOIDX TESTS ###" >> $outfile
echo "###    Author: Patrick Lavin   ###" >> $outfile
echo "##################################" >> $outfile
echo >> $outfile
echo "### Node Info ###" >> $outfile
echo "# host:                       kay ">> $outfile
echo "# node:                       $NODE($HOST)">> $outfile
echo "# arch:                       $ARCH">> $outfile
echo "# mem:                        $MEM">> $outfile
echo "# clock (MHz):                $CLOCK">> $outfile
echo "# ncpus (assigned/available): $ASCPU/$AVCPU">> $outfile
echo >> $outfile
echo "### Run Info ###" >> $outfile
echo "# Runs per config:                 $run*$loop" >> $outfile
echo "# Iterations (num 16 elem copies): $len" >> $outfile
echo >> $outfile
echo "### Compiler Info ###" >> $outfile
echo >> $outfile
echo "# Bandiwdth reported in in MB/s"
echo -e "stride\td\tdelta\t min\t         median\t         max" >> $outfile

for stride in `seq 1 $v`;
do
    echo "running stride $stride"
    for d in `seq 0 $reuse`;
    do
        delta=$((-$d*$stride))
        rm -f tmp$stride.txt
        for i in `seq 1 $run`;
        do
            $SPATTER_BUILD/spatter -R $loop -q --ustride=$stride,$delta --nph -l $len  | grep GATHER | cut -d' ' -f9 >> tmp$stride.txt;
        done

        echo -ne $stride"\t"$d"\t"$delta"\t" >> $outfile
        python3 /home/users/plavin/python/stats.py < tmp$stride.txt >> $outfile
        #sort -r tmp$stride.txt | head -n 1 >> $outfile

        rm -f tmp$stride.txt
    done
    
done

