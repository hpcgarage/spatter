#include <assert.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

// The first byte of trace_entry is an enum
//   0 -> read,  size 4
//   1 -> read,  size 8
//   2 -> write, size 4
//   3 -> write, size 8
//
// The rest of the trace_entry is the address
// 63-56 : rw/sz
// 55-0  : addr

// We use two memory buffers in the kernel, `local` and `mem`
// Each thread is some extra space (GAP_SIZE) in `local` to avoid false sharing.
// The `mem` buffer is large enough for the largest access in the trace.

// Issues:
//  1. Does each thread need only a single 8-byte read-write location?
//      - It probably makes sense to do a small circular buffer so that each
//        thread can immediately issue a write if the preceding access
//        was a read, to avoid stalls
//      - The right amount of space is tricky. Too small and we risk stalls,
//        too large and we risk polluting the cache
//  2. Should we make all writes atomic to deal with write-conflicts?
//  3. Need to ensure that rw_sz() and addr() are inlined
//  4. Need to consider non-temporal stores for the trace

#define GAP_SIZE 32 // 32 doubles == 256 bytes

typedef uint64_t trace_entry;

int rw_sz(trace_entry t) {
    return t >> 56;
}

int addr(trace_entry t) {
    return t & 0xFF'FF'FF'FF'FF'FF'FFLL;
}

void trace_replay_kernel(trace_entry *tr, long len, void *mem, double *local) {

#pragma omp parallel
{
    int tid = omp_get_thread_num();

// nowait ignores all dependencies
#pragma omp for nowait
    for (long i = 0; i < len; i++) {
        switch (rw_sz(tr[i])) {
            case 0:
                local[tid*GAP_SIZE] = ((float*)mem)[addr(tr[i])/4]; // Adjust from address to array-index by dividing by 4
                break;
            case 1:
                local[tid*GAP_SIZE] = ((double*)mem)[addr(tr[i])/8];
                break;
            case 2:
                ((float*)mem)[addr(tr[i])/4] = local[tid*GAP_SIZE];
                break;
            case 3:
                ((double*)mem)[addr(tr[i])/8] = local[tid*GAP_SIZE];
                break;
        }
    }
}

}

// Helpers for creating the trace
long to_rwsz(long long a) {
    return a << 56;
}

long to_addr(long long a) {
    return a & 0xFF'FF'FF'FF'FF'FF'FFLL;
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
