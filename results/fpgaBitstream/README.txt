Some preliminary benchmark tests on the FPGA Bitstream:

See James Wood's notes on the bitstream. It could not handle a length greater than 795 with a delta of 8 (changing the delta and length can increase the max value for either, but there is a cap that is well below what the length should be, 2^24 for example). As a result, this test set is not as robust as I would have liked.

I ran two tests, both employing 50 runs and a pattern of UNIFORM:8:1. One test ran at a length of 512, and the other at 700. I ran 10 tests for each delta that was possible before failing, and calculated the max, min, average, and standard deviation for all data sets.

To repeat the tests I did, simply build the FPGA bitstream directory by first running the configire_sycl_bistream script found in spatter/configure. The source of the script might be incorrect, due to Intel changing its package directory scheme. 
I had to modify /opt/intel/inteloneapi/setvars.sh to /tools/intel/oneapi/1.0/setvars.sh

Note that the environment should be setup properly using the scripts provided on the flubber machines: 
$ . /tools/intel/oneapi/1.0/setvars.sh
$ . /tools/misc/env/cmake_build.sh

cd into the build_sycl_bitstream directory and run make -j20 to build spatter. This might take 1 to 2 hours to build. 

To run the tests, for a length of 512, delta of 4, pattern of UNIFORM:8:1, and 50 runs:
./spatter -l512 --runs=50 -pUNIFORM:8:1 -d4

Please see the documentation in the main spatter directory for more flags and options.

One interesting observation (though I don't know how much relevance that it has) is that higher averages correlate to lower standard deviations. The high values for the runs were essentially the same across all deltas with the same length, but the low values for the runs varied greatly, which contributed to higher standard deviations. The higher the averaage, the less outliers it had, not because it had higher max bandwidths than other runs.
