#include "core/Tensor.h"
#include "core/CheckError.h"
#include <cstring>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>

size_t Tensor::numel() const {
    if (sizes.empty()) return 0;
    return std::accumulate(sizes.begin(), sizes.end(), (size_t)1, std::multiplies<size_t>());
}
void* Tensor::data_ptr() { return storage.raw(); }
const void* Tensor::data_ptr() const { return storage.raw(); }

void Tensor::compute_contiguous_strides() {
    strides.resize(sizes.size());
    if (sizes.empty()) return;
    strides.back() = 1;
    for (int i = (int)sizes.size() - 2; i >= 0; --i) strides[i] = strides[i+1] * sizes[i+1];
}

Tensor Tensor::zeros(const std::vector<int64_t>& sizes, DeviceType dev) {
    Tensor t; t.sizes = sizes; t.device = dev; t.compute_contiguous_strides();
    size_t bytes = t.numel() * sizeof(float);

    t.storage = Storage::create_cpu(bytes);
    std::memset(t.storage.raw(), 0, bytes);
 
    return t;
}

Tensor Tensor::empty(const std::vector<int64_t>& sizes, DeviceType dev) {
    Tensor t; t.sizes = sizes; t.device = dev; t.compute_contiguous_strides();
    size_t bytes = t.numel() * sizeof(float);

    t.storage = Storage::create_cpu(bytes);
    // no memset

    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& sizes, float mean, float std, DeviceType dev) {
    Tensor t = Tensor::zeros(sizes, DeviceType::CPU);
    float* ptr = (float*)t.data_ptr();
    std::random_device rd; std::mt19937 gen(rd()); std::normal_distribution<float> d(mean, std);
    for(size_t i=0; i<t.numel(); ++i) ptr[i] = d(gen);
    
    return t;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    Tensor out = *this;
    std::swap(out.sizes[dim0], out.sizes[dim1]);
    std::swap(out.strides[dim0], out.strides[dim1]);
    return out;
}

Tensor Tensor::flatten() const {
    Tensor out = *this;
    out.sizes = {sizes[0], (int64_t)(numel() / sizes[0])};
    out.compute_contiguous_strides();
    return out;
}

Tensor Tensor::view(const std::vector<int64_t>& new_sizes) const {
    Tensor out = *this;
    out.sizes = new_sizes;
    out.compute_contiguous_strides();
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    int M = sizes[0]; int N = other.sizes[1];
    Tensor out = Tensor::zeros({M, N}, device);
    if (device == DeviceType::CPU) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < sizes[1]; ++k) {
                    const float* a_ptr = (const float*)data_ptr();
                    const float* b_ptr = (const float*)other.data_ptr();
                    sum += a_ptr[i * sizes[1] + k] * b_ptr[k * other.sizes[1] + j];
                }
                float* o_ptr = (float*)out.data_ptr();
                o_ptr[i * N + j] = sum;
            }
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor out = Tensor::zeros(sizes, device);
    const float* a_ptr = (const float*)data_ptr();
    const float* b_ptr = (const float*)other.data_ptr();
    float* o_ptr = (float*)out.data_ptr();
    size_t n = numel();
    for (size_t i = 0; i < n; ++i) {
        o_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return out;
}

void Tensor::ensure_grad() { if(!grad) grad = std::make_shared<Tensor>(Tensor::zeros(sizes, device)); }

void Tensor::zero_grad() { 
    if(grad) {
        std::memset(grad->data_ptr(), 0, grad->numel()*sizeof(float));
    }
}

Tensor Tensor::to(DeviceType target_device) const {
    if (device == target_device) return *this;
    Tensor out = Tensor::zeros(sizes, target_device);
    size_t bytes = numel() * sizeof(float);
    #ifdef USE_CUDA
    if (device == DeviceType::CPU && target_device == DeviceType::CUDA){
        CHECK(cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyHostToDevice));
    } 
    else if (device == DeviceType::CUDA && target_device == DeviceType::CPU) {
        CHECK(cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyDeviceToHost));
    }
    #endif
    return out;
}