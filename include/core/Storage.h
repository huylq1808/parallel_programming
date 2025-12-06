#pragma once
#include <memory>
#include "Device.h"

struct Storage {
    std::shared_ptr<void> data;
    size_t nbytes = 0;
    DeviceType device = DeviceType::CPU;

    static Storage create_cpu(size_t bytes);
    static Storage create_cuda(size_t bytes);
    void* raw() const { return data.get(); }
};