#include "../../include/core/Allocator.h"
#include "../../include/core/CheckError.h"
#include <cstdlib>
#include <stdexcept>

void* cpu_alloc(size_t bytes) {
    void* p = std::malloc(bytes);
    if (!p) throw std::bad_alloc();
    return p;
}
void cpu_free(void* p) { std::free(p); }

#ifdef USE_CUDA
void* cuda_alloc(size_t bytes) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, bytes));
    return p;
}
void cuda_free(void* p) { CHECK(cudaFree(p)); }
#endif