#include <stdio.h>
#include "sycl_kernels.hpp"
#include "../include/parse-args.h"

#include <oneapi/mkl/rng/device.hpp>

#define typedef uint unsigned long

//__device__ int dummy = 0;
sycl::ext::oneapi::experimental::device_global<int> final_block_idx_dev;
sycl::ext::oneapi::experimental::device_global<int> final_thread_idx_dev;
sycl::ext::oneapi::experimental::device_global<int> final_gather_data_dev;

template <int v>
__attribute__((always_inline)) void scatter_t(double *target, double *source, long *ti, long *si, const sycl::nd_item<3> &item, uint8_t *dpct_local)
{
    auto space = (char *)dpct_local;

    int gid = v * item.get_global_id(2);

    double buf[v];
    long idx[v];

    for(int i = 0; i < v; i++){
        buf[i] = source[gid+i];
    }

    for(int i = 0; i < v; i++){
       idx[i] = ti[gid+i];
    }

    for(int i = 0; i < v; i++){
        target[idx[i]] = buf[i];
    }
}

template <int v>
__attribute__((always_inline)) void gather_t(double *target, double *source, long *ti, long *si, const sycl::nd_item<3> &item, uint8_t *dpct_local)
{
    auto space = (char *)dpct_local;

    int gid = v * item.get_global_id(2);
    double buf[v];

    for(int i = 0; i < v; i++){
        buf[i] = source[si[gid+i]];
    }

    for(int i = 0; i < v; i++){
        target[gid+i] = buf[i];
    }

}

//__global__ void gather_new(double *target,

template <int v>
__attribute__((always_inline)) void sg_t(double *target, double *source, long *ti, long *si, const sycl::nd_item<3> &item, uint8_t *dpct_local)
{
    auto space = (char *)dpct_local;

    int gid = v * item.get_global_id(2);
    long sidx[v];
    long tidx[v];

    for(int i = 0; i < v; i++){
        sidx[i] = si[gid+i];
    }
    for(int i = 0; i < v; i++){
        tidx[i] = ti[gid+i];
    }
    for(int i = 0; i < v; i++){
        target[tidx[i]] = source[sidx[i]];
    }

}
#define INSTANTIATE(V)\
template __attribute__((always_inline)) void scatter_t<V>(double* target, double* source, long* ti, long* si, const sycl::nd_item<3> &item, uint8_t *dpct_local);\
template __attribute__((always_inline)) void gather_t<V>(double* target, double* source, long* ti, long* si, const sycl::nd_item<3> &item, uint8_t *dpct_local);\
template __attribute__((always_inline)) void sg_t<V>(double* target, double* source, long* ti, long* si, const sycl::nd_item<3> &item, uint8_t *dpct_local);
INSTANTIATE(1);
INSTANTIATE(2);
INSTANTIATE(4);
INSTANTIATE(5);
INSTANTIATE(8);
INSTANTIATE(16);
INSTANTIATE(32);
INSTANTIATE(64);
INSTANTIATE(128);
INSTANTIATE(256);
INSTANTIATE(512);
INSTANTIATE(1024);
INSTANTIATE(2048);
INSTANTIATE(4096);

int translate_args(unsigned int dim, unsigned int *grid,
		   unsigned int *block, sycl::range<3> *grid_dim,
		   sycl::range<3> *block_dim) {
    if (!grid || !block || dim == 0 || dim > 3) {
        return 1;
    }
    if (dim == 1) {
        *grid_dim = sycl::range<3>(1, 1, grid[0]);
        *block_dim = sycl::range<3>(1, 1, block[0]);
    }else if (dim == 2) {
        *grid_dim = sycl::range<3>(1, grid[1], grid[0]);
        *block_dim = sycl::range<3>(1, block[1], block[0]);
    }else if (dim == 3) {
        *grid_dim = sycl::range<3>(grid[2], grid[1], grid[0]);
        *block_dim = sycl::range<3>(block[2], block[1], block[0]);
    }
    return 0;
}

extern "C" float sycl_sg_wrapper(enum sg_kernel kernel, size_t vector_len,
		      uint dim, uint *grid, uint *block,
		      double *target, double *source, long *ti,
		      long *si, unsigned int shmem, sycl::queue* q) {

    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;

    q->wait();

    auto start = std::chrono::steady_clock::now();
    if(kernel == SCATTER)
    {
        if (vector_len == 1)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<1>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<2>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<4>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 5)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<5>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 8)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<8>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 16)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<16>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 32)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<32>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 64)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<64>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 128)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<128>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 256)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<256>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 512)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<512>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 1024)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<1024>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2048)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<2048>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4096)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_t<4096>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GATHER)
    {
        if (vector_len == 1)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<1>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<2>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<4>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 5)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<5>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 8)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<8>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 16)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<16>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 32)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<32>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 64)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<64>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 128)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<128>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 256)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<256>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 512)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<512>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 1024)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<1024>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2048)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<2048>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4096)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_t<4096>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GS)
    {
        if (vector_len == 1)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<1>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<2>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<4>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 5)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<5>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 8)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<8>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 16)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<16>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 32)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<32>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 64)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<64>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 128)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<128>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 256)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<256>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 512)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<512>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 1024)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<1024>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 2048)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<2048>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else if (vector_len == 4096)
            q->submit([&](sycl::handler &cgh) { sycl::local_accessor<uint8_t, 1> local_mem(sycl::range<1>(shmem), cgh); cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_t<4096>(target, source, ti, si, item, local_mem.get_pointer()); }); });
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else
    {
        printf("ERROR UNRECOGNIZED KERNEL\n");
        exit(1);
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;

}

//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__attribute__((always_inline))
void scatter_block(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

#ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = blockIdx.x;
        final_thread_idx_dev = threadIdx.x;
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;

    //for (int i = 0; i < wpb; i++) {
        src_loc[idx_shared[tid%V]] = idx_shared[tid%V];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}
}

//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__attribute__((always_inline))
void scatter_block_random(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);
    int ngatherperblock = item.get_local_range(2) / V;
    int gatherid = tid / V;

    unsigned long long sequence = item.get_group(2); // all thread blocks can use same sequence
    unsigned long long offset = gatherid;

    oneapi::mkl::rng::device::philox4x32x10<> engine(seed, item.get_local_id(2));
    oneapi::mkl::rng::device::uniform<> distr;
    int random_gatherid = oneapi::mkl::rng::device::generate(distr, engine);

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }


    double *src_loc = src + (bid*ngatherperblock+random_gatherid)*delta;

    //for (int i = 0; i < wpb; i++) {
        src_loc[idx_shared[tid%V]] = idx_shared[tid%V];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}
}

//V2 = 8
//assume block size >= index buffer size
//assume index buffer size divides block size
template<int V>
__attribute__((always_inline))
void gather_block(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

#ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;
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

template<int V>
__attribute__((always_inline))
void gather_block_morton(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, uint32_t *order, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+order[gatherid])*delta;

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

template<int V>
__attribute__((always_inline))
void gather_block_stride(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, int stride, char validate, const sycl::nd_item<3> &item)
{
    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

#ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    int ngatherperblock = item.get_local_range(2) / V;
    int gatherid = tid / V;

    double *src_loc = src + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = src_loc[stride*(tid%V)];
    }
    #endif

    double x;

    //for (int i = 0; i < wpb; i++) {
    x = src_loc[stride*(tid%V)];
    //src_loc[idx_shared[tid%V]] = 1337.;
    //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

template<int V>
__attribute__((always_inline))
void gather_block_random(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);
    int ngatherperblock = item.get_local_range(2) / V;
    int gatherid = tid / V;

    unsigned long long sequence = item.get_group(2); // all thread blocks can use same sequence
    unsigned long long offset = gatherid;

    oneapi::mkl::rng::device::philox4x32x10<> engine(seed, item.get_local_id(2));
    oneapi::mkl::rng::device::uniform<> distr;
    int random_gatherid = oneapi::mkl::rng::device::generate(distr, engine);

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    double *src_loc = src + (bid*ngatherperblock+random_gatherid)*delta;
    double x;

    //for (int i = 0; i < wpb; i++) {
        x = src_loc[idx_shared[tid%V]];
        //src_loc[idx_shared[tid%V]] = 1337.;
        //src_loc += delta;
    //}

    if (x==0.5) src[0] = x;

}

//todo -- add WRAP
template <int V>
__attribute__((always_inline))
void gather_new(double *source, sgIdx_t *idx, size_t delta, int dummy, int wpt, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    //int bid  = item.get_group(2);
    //int nblk = blockDim.x;

    if (tid < V) {
        idx_shared[tid] = idx[tid];
    }

    int gid = item.get_global_id(2);
    double *sl = source + wpt*gid*delta;

    double buf[V];

    for (int j = 0; j < wpt; j++) {
        for (int i = 0; i < V; i++) {
            buf[i] = sl[idx_shared[i]];
            //source[i+gid*delta] = 8;
            //sl[i] = sl[idx[i]];
        }
        sl = sl + delta;
    }

    if (dummy) {
        sl[idx_shared[0]] = buf[dummy];
    }

    /*
    for (int i = 0; i < V; i++) {
        if (buf[i] == 199402) {
            printf("oop\n");
        }
    }
    */

        //printf("idx[1]: %d\n", idx[1]);
        /*
        for (int i = 0; i < V; i++) {
            printf("idx %d is %zu", i, idx[i]);
        }
        printf("\n");
        */

}

#define INSTANTIATE2(V)\
template __attribute__((always_inline)) void gather_new<V>(double* source, sgIdx_t* idx, size_t delta, int dummy, int wpt, const sycl::nd_item<3> &item); \
template __attribute__((always_inline)) void gather_block<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate, const sycl::nd_item<3> &item);\
template __attribute__((always_inline)) void gather_block_morton<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, uint32_t *order, char validate, const sycl::nd_item<3> &item);\
template __attribute__((always_inline)) void gather_block_stride<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, int stride, char validate, const sycl::nd_item<3> &item);\
template __attribute__((always_inline)) void scatter_block<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, char validate, const sycl::nd_item<3> &item); \
template __attribute__((always_inline)) void gather_block_random<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n, const sycl::nd_item<3> &item); \
template __attribute__((always_inline)) void scatter_block_random<V>(double *src, ssize_t* idx, size_t idx_len, size_t delta, int wpb, size_t seed, size_t n, const sycl::nd_item<3> &item);

//INSTANTIATE2(1);
//INSTANTIATE2(2);
//INSTANTIATE2(4);
//INSTANTIATE2(5);
INSTANTIATE2(8);
INSTANTIATE2(16);
INSTANTIATE2(32);
INSTANTIATE2(64);
INSTANTIATE2(73);
INSTANTIATE2(128);
INSTANTIATE2(256);
INSTANTIATE2(512);
INSTANTIATE2(1024);
INSTANTIATE2(2048);
INSTANTIATE2(4096);

extern "C" float sycl_block_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        ssize_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap,
        int wpt,
        size_t morton,
        uint32_t *order,
        uint32_t *order_dev,
        int stride,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(pat_dev, pat, sizeof(sgIdx_t) * pat_len).wait();
    q->memcpy(order_dev, order, sizeof(uint32_t) * n).wait();

    q->wait();
    auto start = std::chrono::steady_clock::now();

    // KERNEL
    if (kernel == GATHER) {
        if (morton) {
            if (pat_len == 8) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<8>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 16) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<16>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 32) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<32>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 64) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<64>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 73) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<73>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 128) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<128>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 256) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<256>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 512) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<512>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 1024) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<1024>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 2048) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<2048>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else if (pat_len == 4096) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_morton<4096>(source, pat_dev, pat_len, delta, wpt, order_dev, validate, item); });
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }
        } else if (stride >= 0) {
            if (pat_len == 8) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<8>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 16) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<16>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 32) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<32>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 64) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<64>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 73) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<73>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 128) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<128>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 256) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<256>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 512) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<512>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 1024) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<1024>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 2048) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<2048>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else if (pat_len == 4096) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_stride<4096>(source, pat_dev, pat_len, delta, wpt, stride, validate, item); });
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }

        } else {
            if (pat_len == 8) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<8>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 16) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<16>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 32) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<32>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 64) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<64>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 73) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<73>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 128) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<128>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 256) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<256>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 512) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<512>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 1024) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<1024>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 2048) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<2048>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else if (pat_len == 4096) {
                q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block<4096>(source, pat_dev, pat_len, delta, wpt, validate, item); });
            }else {
                printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
                exit(1);
            }
        }
	q->memcpy(final_gather_data, final_gather_data_dev, sizeof(double)); // cudaMemcpyDeviceToHost);
    } else if (kernel == SCATTER) {
        if (pat_len == 8) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<8>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 16) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<16>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 32) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<32>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 64) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<64>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 128) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<128>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len ==256) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<256>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 512) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<512>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 1024) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<1024>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len == 2048) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<2048>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else if (pat_len ==4096) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block<4096>(source, pat_dev, pat_len, delta, wpt, validate, item); });
        }else {
            printf("ERROR NOT SUPPORTED, %zu\n", pat_len);
            exit(1);
        }

    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    q->memcpy(final_block_idx, final_block_idx_dev, sizeof(int)).wait();
    q->memcpy(final_thread_idx, final_thread_idx_dev, sizeof(int)).wait();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;


}

extern "C" float sycl_block_random_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        ssize_t* pat_dev,
        ssize_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap,
        int wpt, size_t seed, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(pat_dev, pat, sizeof(sgIdx_t) * pat_len).wait();

    q->wait();
    auto start = std::chrono::steady_clock::now();
    // KERNEL
    if (kernel == GATHER) {
        if (pat_len == 8) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<8>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 16) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<16>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 32) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<32>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 64) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<64>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 128) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<128>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len ==256) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<256>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 512) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<512>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 1024) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<1024>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 2048) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<2048>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len ==4096) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_block_random<4096>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else {
            printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
        }
    } else if (kernel == SCATTER) {
        if (pat_len == 8) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<8>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 16) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<16>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 32) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<32>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 64) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<64>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 128) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<128>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len ==256) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<256>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 512) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<512>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 1024) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<1024>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len == 2048) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<2048>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else if (pat_len ==4096) {
            q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { scatter_block_random<4096>(source, pat_dev, pat_len, delta, wpt, seed, n, item); });
        }else {
            printf("ERROR NOT SUPPORTED, %zu\n", pat_len);
        }
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;


}

extern "C" float sycl_new_wrapper(uint dim, uint* grid, uint* block,
        enum sg_kernel kernel,
        double *source,
        sgIdx_t* pat_dev,
        sgIdx_t* pat,
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t wrap,
        int wpt, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(pat_dev, pat, sizeof(sgIdx_t) * pat_len).wait();

    q->wait();
    auto start = std::chrono::steady_clock::now();

    // KERNEL
    if (pat_len == 8) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<8>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 16) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<16>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 64) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<64>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 256) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<256>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 512) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<512>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 1024) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<1024>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 2048) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<2048>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else if (pat_len == 4096) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { gather_new<4096>(source, pat_dev, (long)delta, 0, wpt, item); });
    }else {
        printf("ERROR NOT SUPPORTED\n");
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;

}
/*
    dim3 grid_dim, block_dim;
    syclEvent_t start, stop;

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;

    syclEventCreate(&start);
    syclEventCreate(&stop);

    syclDeviceSynchronize();
    syclEventRecord(start);
    if(kernel == SCATTER)
    {
        if (vector_len == 1)
            scatter_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            scatter_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            scatter_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            scatter_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            scatter_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            scatter_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            scatter_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            scatter_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GATHER)
    {
        if (vector_len == 1)
            gather_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            gather_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            gather_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            gather_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            gather_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            gather_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            gather_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            gather_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else if(kernel == GS)
    {
        if (vector_len == 1)
            sg_t<1><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 2)
            sg_t<2><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 4)
            sg_t<4><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 5)
            sg_t<5><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 8)
            sg_t<8><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 16)
            sg_t<16><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 32)
            sg_t<32><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else if (vector_len == 64)
            sg_t<64><<<grid_dim,block_dim,shmem>>>(target, source, ti, si);
        else
        {
            printf("ERROR: UNSUPPORTED VECTOR LENGTH\n");
            exit(1);
        }
    }
    else
    {
        printf("ERROR UNRECOGNIZED KERNEL\n");
        exit(1);
    }
    syclEventRecord(stop);
    syclEventSynchronize(stop);

    float time_ms = 0;
    syclEventElapsedTime(&time_ms, start, stop);
    return time_ms;

}*/

template<int V>
__attribute__((always_inline))
void sg_block(double *source, double* target, sgIdx_t* pat_gath, sgIdx_t* pat_scat, spSize_t pat_len, size_t delta_gather, size_t delta_scatter, int wpt, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx_gath = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);
    tile_t& idx_scat = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

#ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    if (tid < V) {
        idx_gath[tid] = pat_gath[tid];
        idx_scat[tid] = pat_scat[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;
    int nscatterperblock = ngatherperblock;

    int gatherid = tid / V;
    int scatterid = gatherid;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta_gather;
    double *target_loc = target + (bid*nscatterperblock+scatterid)*delta_scatter;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[idx_gath[tid%V]];
        return;
    }
    #endif

    target_loc[idx_scat[tid%V]] = source_loc[idx_gath[tid%V]];
}

#define INSTANTIATE3(V)\
template __attribute__((always_inline)) void sg_block<V>(double* source, double* target, sgIdx_t* pat_gath, sgIdx_t* pat_scat, spSize_t pat_len, size_t delta_gather, size_t delta_scatter, int wpt, char validate, const sycl::nd_item<3> &item);

//INSTANTIATE3(1);
//INSTANTIATE3(2);
//INSTANTIATE3(4);
//INSTANTIATE3(5);
INSTANTIATE3(8);
INSTANTIATE3(16);
INSTANTIATE3(32);
INSTANTIATE3(64);
INSTANTIATE3(73);
INSTANTIATE3(128);
INSTANTIATE3(256);
INSTANTIATE3(512);
INSTANTIATE3(1024);
INSTANTIATE3(2048);
INSTANTIATE3(4096);

extern "C" float sycl_block_sg_wrapper(uint dim, uint* grid, uint* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* pat_gath_dev,
        sgIdx_t* pat_scat_dev,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
	char validate, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(pat_gath_dev, rc->pattern_gather, sizeof(sgIdx_t) * rc->pattern_gather_len).wait();
    q->memcpy(pat_scat_dev, rc->pattern_scatter, sizeof(sgIdx_t) * rc->pattern_scatter_len).wait();

    size_t delta_gather = rc->delta_gather;
    size_t delta_scatter = rc->delta_scatter;

    spSize_t pat_len = rc->pattern_gather_len;

    q->wait();
    auto start = std::chrono::steady_clock::now();

    // KERNEL
    if (pat_len == 8) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<8>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 16) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<16>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 32) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<32>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 64) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<64>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 73) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<73>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 128) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<128>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 256) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<256>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 512) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<512>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 1024) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<1024>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 2048) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<2048>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else if (pat_len == 4096) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { sg_block<4096>(source, target, pat_gath_dev, pat_scat_dev, pat_len, delta_gather, delta_scatter, wpt, validate, item); });
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    q->memcpy(final_block_idx, final_block_idx_dev, sizeof(int)).wait();
    q->memcpy(final_thread_idx, final_thread_idx_dev, sizeof(int)).wait();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;
}


template<int V>
__attribute__((always_inline))
void multiscatter_block(double *source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);
    tile_t& idx_scat = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

#ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    if (tid < V) {
        idx[tid] = outer_pat[tid];
        idx_scat[tid] = inner_pat[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;

    int gatherid = tid / V;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[tid%V];
        return;
    }
    #endif

    source_loc[idx[idx_scat[tid%V]]] = target[tid%V];
}

#define INSTANTIATE4(V)\
template __attribute__((always_inline)) void multiscatter_block<V>(double* source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate, const sycl::nd_item<3> &item);

//INSTANTIATE4(1);
//INSTANTIATE4(2);
//INSTANTIATE4(4);
//INSTANTIATE4(5);
INSTANTIATE4(8);
INSTANTIATE4(16);
INSTANTIATE4(32);
INSTANTIATE4(64);
INSTANTIATE4(73);
INSTANTIATE4(128);
INSTANTIATE4(256);
INSTANTIATE4(512);
INSTANTIATE4(1024);
INSTANTIATE4(2048);
INSTANTIATE4(4096);

extern "C" float sycl_block_multiscatter_wrapper(uint dim, uint* grid, uint* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(outer_pat, rc->pattern, sizeof(sgIdx_t) * rc->pattern_len).wait();
    q->memcpy(inner_pat, rc->pattern_scatter, sizeof(sgIdx_t) * rc->pattern_scatter_len).wait();

    size_t delta = rc->delta;

    spSize_t pat_len = rc->pattern_scatter_len;

    q->wait();
    auto start = std::chrono::steady_clock::now();

    // KERNEL
    if (pat_len == 8) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<8>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 16) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<16>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 32) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<32>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 64) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<64>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 73) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<73>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 128) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<128>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 256) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<256>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 512) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<512>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 1024) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<1024>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 2048) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<2048>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 4096) {
        q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) { multiscatter_block<4096>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    q->memcpy(final_block_idx, final_block_idx_dev, sizeof(int)).wait();
    q->memcpy(final_thread_idx, final_thread_idx_dev, sizeof(int)).wait();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;
}

template<int V>
__attribute__((always_inline))
void multigather_block(double *source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate, const sycl::nd_item<3> &item)
{
    sycl::group work_grp = item.get_group();
    using tile_t = int[V];
    tile_t& idx = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);
    tile_t& idx_gath = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(work_grp);

    int tid = item.get_local_id(2);
    int bid = item.get_group(2);

    #ifdef VALIDATE
    if (validate) {
        final_block_idx_dev = item.get_group(2);
        final_thread_idx_dev = item.get_local_id(2);
    }
    #endif

    if (tid < V) {
        idx[tid] = outer_pat[tid];
        idx_gath[tid] = inner_pat[tid];
    }

    int ngatherperblock = item.get_local_range(2) / V;

    int gatherid = tid / V;

    double *source_loc = source + (bid*ngatherperblock+gatherid)*delta;

    #ifdef VALIDATE
    if (validate) {
        final_gather_data_dev = source_loc[idx[idx_gath[tid%V]]];
        return;
    }
    #endif

    target[tid%V] = source_loc[idx[idx_gath[tid%V]]];
}

#define INSTANTIATE5(V)\
template __attribute__((always_inline)) void multigather_block<V>(double* source, double* target, sgIdx_t* outer_pat, sgIdx_t* inner_pat, spSize_t pat_len, size_t delta, int wpt, char validate, const sycl::nd_item<3> &item);

//INSTANTIATE5(1);
//INSTANTIATE5(2);
//INSTANTIATE5(4);
//INSTANTIATE5(5);
INSTANTIATE5(8);
INSTANTIATE5(16);
INSTANTIATE5(32);
INSTANTIATE5(64);
INSTANTIATE5(73);
INSTANTIATE5(128);
INSTANTIATE5(256);
INSTANTIATE5(512);
INSTANTIATE5(1024);
INSTANTIATE5(2048);
INSTANTIATE5(4096);

extern "C" float sycl_block_multigather_wrapper(uint dim, uint* grid, uint* block,
        double *source,
        double *target,
        struct run_config* rc,
        sgIdx_t* outer_pat,
        sgIdx_t* inner_pat,
        int wpt,
        int *final_block_idx,
        int *final_thread_idx,
        double *final_gather_data,
        char validate, sycl::queue* q)
{
    sycl::range<3> grid_dim(1, 1, 1), block_dim(1, 1, 1);

    if(translate_args(dim, grid, block, &grid_dim, &block_dim)) return 0;
    q->memcpy(outer_pat, rc->pattern, sizeof(sgIdx_t) * rc->pattern_len).wait();
    q->memcpy(inner_pat, rc->pattern_gather, sizeof(sgIdx_t) * rc->pattern_gather_len).wait();

    size_t delta = rc->delta;

    spSize_t pat_len = rc->pattern_gather_len;

    q->wait();
    auto start = std::chrono::steady_clock::now();

    // KERNEL
    if (pat_len == 8) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<8>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 16) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<16>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 32) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<32>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 64) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<64>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 73) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<73>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 128) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<128>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 256) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<256>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 512) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<512>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 1024) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<1024>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 2048) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<2048>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else if (pat_len == 4096) {
	q->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> item) {
		multigather_block<4096>(source, target, outer_pat, inner_pat, pat_len, delta, wpt, validate, item); });
    }else {
        printf("ERROR NOT SUPPORTED: %zu\n", pat_len);
    }

    q->wait();
    auto stop = std::chrono::steady_clock::now();

    q->memcpy(final_block_idx, final_block_idx_dev, sizeof(int)).wait();
    q->memcpy(final_thread_idx, final_thread_idx_dev, sizeof(int)).wait();

    float time_ms = 0;
    time_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return time_ms;
}
