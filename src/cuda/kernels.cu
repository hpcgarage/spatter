template<int N, typename T>
__global__
void my_kernel(T* data) {
    T data0 = data[0];
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
};

template<V>
__global__ void gather(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate, const int VV)
{
    __shared__ ssize_t idx_shared[1024];
    //extern __shared__ ssize_t idx_shared[];

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    //if (tid < V) {
        idx_shared[tid] = idx[tid];
    //}

    int ngatherperblock = blockDim.x / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = src_loc[idx_shared[tid%V]];
        return;
    }
    #endif

    double x;

    //for (int i = 0; i < wpb; i++) {
        x = src_loc[idx_shared[tid%V]];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}
