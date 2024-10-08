#include <assert.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include "trace-replay-kernel.h"

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

#define DEBUG 0
#define GAP_SIZE 32 // 32 doubles == 256 bytes

typedef uint64_t trace_entry;

int rw_sz(trace_entry t) {
    return t >> ADDR_BITS;
}

long long addr(trace_entry t) {
    return t & ADDR_MASK;
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
#if DEBUG
                printf("rd 4 0x%llx\n", addr(tr[i]));
#endif
                local[tid*GAP_SIZE] = ((float*)mem)[addr(tr[i])/4]; // Adjust from address to array-index by dividing by 4
                break;
            case 1:
#if DEBUG
                printf("rd 8 0x%llx\n", addr(tr[i]));
#endif
                local[tid*GAP_SIZE] = ((double*)mem)[addr(tr[i])/8];
                break;
            case 2:
#if DEBUG
                printf("wr 4 0x%llx\n", addr(tr[i]));
#endif
                ((float*)mem)[addr(tr[i])/4] = local[tid*GAP_SIZE];
                break;
            case 3:
#if DEBUG
                printf("wr 8 0x%llx\n", addr(tr[i]));
#endif
                ((double*)mem)[addr(tr[i])/8] = local[tid*GAP_SIZE];
                break;
        }
    }
}

}
