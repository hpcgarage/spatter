#ifndef CUDA_JIT_BACKEND_H
#define CUDA_JIT_BACKEND_H
#include <stdint.h>
#include "../include/parse-args.h"
#include "sgbuf.h"

#pragma once
#ifdef __cplusplus
extern "C"
{
#endif
extern void create_dev_buffers_cuda(sgDataBuf* source);
extern int find_device_cuda(char *name);
extern double cuda_jit_wrapper(struct run_config rc);
#ifdef __cplusplus
}
#endif
#endif

