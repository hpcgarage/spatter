#ifndef TRACE_REPLAY_KERNEL_H
#define TRACE_REPLAY_KERNEL_H
#include <cstdint>

#define ADDR_BITS 56
#define ADDR_MASK 0xFF'FF'FF'FF'FF'FF'FFLL
#define GAP_SIZE 32 // 32 doubles == 256 bytes
typedef uint64_t trace_entry;
void trace_replay_kernel(trace_entry *tr, long len, void *mem, double *local);

#endif
