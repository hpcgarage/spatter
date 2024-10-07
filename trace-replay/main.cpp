#include <assert.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include "trace-replay-kernel.h"

// Helpers for creating the trace
long to_rwsz(long long a) {
    return a << ADDR_BITS;
}

long to_addr(long long a) {
    return a & ADDR_MASK;
}

int main() {
    assert(sizeof(long) == 8);

    // Create a trace with 8 entries
    trace_entry *tr = (trace_entry*)malloc(sizeof(trace_entry) * 8);

    tr[0] = to_rwsz(0) | to_addr(0x0);  // Read  4
    tr[1] = to_rwsz(1) | to_addr(0x8);  // Read  8
    tr[2] = to_rwsz(2) | to_addr(0xC);  // Write 4
    tr[3] = to_rwsz(3) | to_addr(0x18); // Write 8

    tr[4] = to_rwsz(3) | to_addr(0x30); // Write 8
    tr[5] = to_rwsz(2) | to_addr(0x34); // Write 4
    tr[6] = to_rwsz(1) | to_addr(0x40); // Read  8
    tr[7] = to_rwsz(0) | to_addr(0x44); // Read  4

    double *local = (double*)malloc(sizeof(double) * GAP_SIZE * omp_get_max_threads());
    double *mem   = (double*)malloc(sizeof(char)   * 0x64);

    trace_replay_kernel(tr, 8, mem, local);
    printf("All done!\n");
}
