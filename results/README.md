The result format is as follows:

```
backend kernel op time source_size target_size idx_size worksets bytes_moved usable_bandwidth omp_threads vector_len

backend:
    CUDA, OPENMP, or OPENCL

kernel:
    SCATTER, GATHER, or SG

op:
    COPY, or ACCUM

time:
    The time (in seconds) that the kernel took to execute

source_size, target_size, and idx_size:
     These measure the size (in bytes) of the the buffers used for a single run of the kernel (i.e. if multiple worksets are used, it is not reflected here)

worksets:
    The number of worksets (copies of the source, target, and index buffers) allocated

bytes_moved:
    The number of data bytes moved during a run of the kernel

usabled_bandwidth:
    This bandwidth does not account for bandwidth spend on index buffers. It is bytes_moved/time/1024/1024, which is MiB/s
     
omp_threads:
    Meaningful only if using OpenMP backend

vector_len:
    The number of elements copied by a single work item (OpenCL) or thread (CUDA)  
```
