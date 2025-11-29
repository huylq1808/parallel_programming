#pragma once
#include <cstddef>

void* cpu_alloc(size_t bytes);
void cpu_free(void* p);

#ifdef USE_CUDA
void* cuda_alloc(size_t bytes);
void cuda_free(void* p);
#endif