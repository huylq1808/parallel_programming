#include "core/Storage.h"
#include "core/Allocator.h"
#include <stdexcept>

Storage Storage::create_cpu(size_t bytes) {
    Storage s;
    s.data = std::shared_ptr<void>(cpu_alloc(bytes), [](void* p){ cpu_free(p); });
    s.nbytes = bytes;
    s.device = DeviceType::CPU;
    return s;
}

Storage Storage::create_cuda(size_t bytes) {
    Storage s;
    #ifdef USE_CUDA
    s.data = std::shared_ptr<void>(cuda_alloc(bytes), [](void* p){ cuda_free(p); });
    s.device = DeviceType::CUDA;
    #else
    throw std::runtime_error("CUDA not enabled");
    #endif
    s.nbytes = bytes;
    return s;
}