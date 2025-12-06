#include "core/Tensor.h"
#include "core/CheckError.h"
#include <cstring>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>


// --- Kernels (Naive Implementation) ---
__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


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
    if (dev == DeviceType::CPU) {
        t.storage = Storage::create_cpu(bytes);
        std::memset(t.storage.raw(), 0, bytes);
    } 
    #ifdef USE_CUDA
    else {
        t.storage = Storage::create_cuda(bytes);
        CHECK(cudaMemset(t.storage.raw(), 0, bytes));
    }
    #endif
    return t;
}

Tensor Tensor::empty(const std::vector<int64_t>& sizes, DeviceType dev) {
    Tensor t; t.sizes = sizes; t.device = dev; t.compute_contiguous_strides();
    size_t bytes = t.numel() * sizeof(float);
    if (dev == DeviceType::CPU) {
        t.storage = Storage::create_cpu(bytes);
        // no memset
    } 
    #ifdef USE_CUDA
    else {
        t.storage = Storage::create_cuda(bytes);
        // no cudaMemset
    }
    #endif
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& sizes, float mean, float std, DeviceType dev) {
    Tensor t = Tensor::zeros(sizes, DeviceType::CPU);
    float* ptr = (float*)t.data_ptr();
    std::random_device rd; std::mt19937 gen(rd()); std::normal_distribution<float> d(mean, std);
    for(size_t i=0; i<t.numel(); ++i) ptr[i] = d(gen);
    
    if (dev == DeviceType::CUDA) {
        // Simple Copy for initialization
        Tensor gpu_t = Tensor::zeros(sizes, DeviceType::CUDA);
        #ifdef USE_CUDA
        CHECK(cudaMemcpy(gpu_t.data_ptr(), t.data_ptr(), t.numel()*sizeof(float), cudaMemcpyHostToDevice));
        #endif
        return gpu_t;
    }
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
        const float* a_ptr = (const float*)data_ptr();
        const float* b_ptr = (const float*)other.data_ptr();
        float* o_ptr = (float*)out.data_ptr();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < sizes[1]; ++k) {
                    sum += a_ptr[i * sizes[1] + k] * b_ptr[k * N + j];
                }
                o_ptr[i * N + j] = sum;
            }
        }
    }
    #ifdef USE_CUDA
    else {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
        matmul_kernel<<<gridSize, blockSize>>>(
            (const float*)data_ptr(),
            (const float*)other.data_ptr(),
            (float*)out.data_ptr(),
            M, sizes[1], N
        );
        CHECK(cudaDeviceSynchronize());

    }
    #endif
    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor out = Tensor::zeros(sizes, device);
    if (device == DeviceType::CPU) {
        // Bạn cần khai báo hàm này trong Ops.h và implement trong ops_cpu.cpp
        const float* a_ptr = (const float*)data_ptr();
        const float* b_ptr = (const float*)other.data_ptr();
        float* o_ptr = (float*)out.data_ptr();
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            o_ptr[i] = a_ptr[i] + b_ptr[i];
        }

    }
    #ifdef USE_CUDA
    else {
        // Bạn cần khai báo hàm này trong Ops.h và implement trong ops_bridge.cu
        size_t n = numel();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        add_kernel<<<numBlocks, blockSize>>>(
            (const float*)data_ptr(),
            (const float*)other.data_ptr(),
            (float*)out.data_ptr(),
            n
        );
        CHECK(cudaDeviceSynchronize());
    }
    #endif

    return out;
}

void Tensor::ensure_grad() { if(!grad) grad = std::make_shared<Tensor>(Tensor::zeros(sizes, device)); }

void Tensor::zero_grad() { 
    if(grad) {
        if(device==DeviceType::CPU) std::memset(grad->data_ptr(), 0, grad->numel()*sizeof(float));
        #ifdef USE_CUDA
        else CHECK(cudaMemset(grad->data_ptr(), 0, grad->numel()*sizeof(float)));
        #endif
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