#!/bin/bash
#This script runs the uniform stride and random access tests used to generate the plots in Figures 3,4, and 7.

export OMP_PROC_BIND=master
export OMP_WAIT_POLICY=active
export OMP_NUM_THREADS=12
export OMP_PLACES=sockets

#Specify the name for your system and CPU/accelerator type; this is used to sort results
LONGNAME=skylar-skl
SHORTNAME=skl

###########Run the test scripts


# -m 1 is required with Cori's flat mode to specify to use MCDRAM as the preferred domain
#Run STREAM
numactl -N 0 -l ./run-stream.sh $LONGNAME
numactl -N 0 -l ./sg-sparse-roofline.sh openmp $SHORTNAME
numactl -N 0 -l ./sg-rdm-roofline.sh openmp $SHORTNAME
#Sort results
./organize-results.sh openmp $LONGNAME $SHORTNAME
