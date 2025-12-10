#include "core/Allocator.h"
#include "core/CheckError.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>

void* cpu_alloc(size_t bytes) { return std::malloc(bytes); }
void cpu_free(void* p) { std::free(p); }

#ifdef USE_CUDA

class CudaMemoryPool {
private:
    std::unordered_map<size_t, std::vector<void*>> free_blocks;  // bucket -> free pointers
    std::unordered_map<void*, size_t> alloc_sizes;               // ptr -> bucket size
    std::mutex mtx;
    
    size_t roundUpToBucket(size_t bytes) {
        if (bytes <= 256) return 256;
        size_t power = 256;
        while (power < bytes) power <<= 1;
        return power;
    }

public:
    static CudaMemoryPool& instance() {
        static CudaMemoryPool pool;
        return pool;
    }
    
    void* allocate(size_t bytes) {
        size_t bucket = roundUpToBucket(bytes);
        
        std::lock_guard<std::mutex> lock(mtx);
        
        if (!free_blocks[bucket].empty()) {
            void* ptr = free_blocks[bucket].back();
            free_blocks[bucket].pop_back();
            alloc_sizes[ptr] = bucket;  // Track size
            return ptr;
        }
        
        void* ptr = nullptr;
        CHECK(cudaMalloc(&ptr, bucket));
        alloc_sizes[ptr] = bucket;  // Track size
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mtx);
        
        auto it = alloc_sizes.find(ptr);
        if (it == alloc_sizes.end()) {
            // Unknown allocation, just free it
            cudaFree(ptr);
            return;
        }
        
        size_t bucket = it->second;
        alloc_sizes.erase(it);
        free_blocks[bucket].push_back(ptr);  
    }
    
    ~CudaMemoryPool() {
        for (auto& [bucket, blocks] : free_blocks) {
            for (void* ptr : blocks) {
                cudaFree(ptr);
            }
        }
    }
};

void* cuda_alloc(size_t bytes) {
    return CudaMemoryPool::instance().allocate(bytes);
}

void cuda_free(void* p) {
    CudaMemoryPool::instance().deallocate(p);
}

#endif