#!/bin/bash
#This script runs the uniform stride and random access tests used to generate the plots in Figures 3,4, and 7.

#One socket is used on the P9, as with other systems

#Each thread is exposed as a "thread", so running normally will lead to a warning about OMP_PROC_BIND
export OMP_PROC_BIND=TRUE
export OMP_WAIT_POLICY=active
export OMP_NUM_THREADS=64
export OMP_PLACES={0:64}

#Specify the name for your system and CPU/accelerator type; this is used to sort results
LONGNAME=newell-pwr9
SHORTNAME=pwr9

#Run the test scripts


#Run STREAM
#numactl -N 0 -l ./run-stream.sh $LONGNAME
#numactl -N 0 -l ./sg-sparse-roofline.sh openmp $SHORTNAME
#numactl -N 0 -l ./sg-rdm-roofline.sh openmp $SHORTNAME
#Sort results
./organize-results.sh openmp $SHORTNAME cpu
