#pragma once
#include <vector>
#include <memory>
#include "Storage.h"

struct Tensor {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    DeviceType device = DeviceType::CPU;
    Storage storage;
    
    std::shared_ptr<Tensor> grad = nullptr;
    bool requires_grad = false;

    Tensor() = default;

    // Factories
    static Tensor zeros(const std::vector<int64_t>& sizes, DeviceType dev = DeviceType::CPU);
    static Tensor randn(const std::vector<int64_t>& sizes, float mean = 0.0f, float std = 0.08f, DeviceType dev = DeviceType::CPU);
    static Tensor empty(const std::vector<int64_t>& sizes, DeviceType dev = DeviceType::CPU);
    
    // Helpers
    size_t numel() const;
    void* data_ptr();
    const void* data_ptr() const;
    void compute_contiguous_strides();

    // Ops
    Tensor matmul(const Tensor& other) const;
    Tensor add(const Tensor& other) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor flatten() const;
    Tensor view(const std::vector<int64_t>& new_sizes) const;

    // Autograd Utils
    void ensure_grad();
    void zero_grad();
    Tensor to(DeviceType target_device) const; // Copy CPU <-> GPU
};