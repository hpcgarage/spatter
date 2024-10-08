#include <assert.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include "trace-replay-kernel.h"

// Helpers for creating the trace
long to_rwsz(long long a) {
    return a << ADDR_BITS;
}

long to_addr(long long a) {
    return a & ADDR_MASK;
}

int main() {
    assert(sizeof(long long) == 8);

    // Create a trace with 8 entries
    long long target_size = 16LL * 1024 * 1024 * 1024; // 16 GiB, read from first half, write to second

    // Each trace entry will read 8 bytes then write 8 bytes
    // Need a trace entry for each of the 1024*1024*1024 locations, and one for read, one for write
    trace_entry *tr = (trace_entry*)malloc(sizeof(trace_entry) * 1024LL*1024*1024 * 2);

    for (long i = 0; i < 1024LL*1024*1024; i++) {
        tr[2*i  ] = to_rwsz(1) | to_addr(8*i);
        tr[2*i+1] = to_rwsz(3) | to_addr(8*i + 8LL*1024*1024*1024);
    }

    double *local = (double*)malloc(sizeof(double) * GAP_SIZE * omp_get_max_threads());
    void *mem   = (void*)malloc(sizeof(char)   * target_size);

    auto start = std::chrono::high_resolution_clock::now();
    trace_replay_kernel(tr, 1024LL*1024*1024*2, mem, local);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    long long size_mb = target_size / (1024*1024);
    double time_seconds = elapsed.count();
    double bw_mbps = size_mb / elapsed.count();
    printf("threads size_mb time_s bandwidth_mb\n");
    printf("%d %lld %.2lf %.2lf\n", omp_get_max_threads(), size_mb, time_seconds, bw_mbps);
}
