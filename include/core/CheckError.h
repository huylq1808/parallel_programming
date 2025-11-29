#pragma once
#include <cstdio>
#include <cstdlib>

// Macro check error for cuda calling 
#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}
#else
#define CHECK(call) call
#endif