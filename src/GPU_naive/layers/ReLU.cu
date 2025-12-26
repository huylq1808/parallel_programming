#include "layers/ReLU.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {
__global__ void k_relu_fwd(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

__global__ void k_relu_bwd(const float* in, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
}
} // namespace

Tensor ReLU::forward(const Tensor& input) {
    // 1. Chuyển input sang GPU (nếu chưa có) và lưu vào cache
    input_cache = input.to(DeviceType::CUDA); 
    
    // 2. Tạo output mới trên GPU
    Tensor out = Tensor::empty(input_cache.sizes, DeviceType::CUDA);

    int n = input_cache.numel();
    if (n == 0) return input_cache;

    // 3. QUAN TRỌNG: Lấy pointer từ input_cache (đảm bảo là GPU pointer)
    const float* in_ptr = (const float*)input_cache.data_ptr();
    float* out_ptr = (float*)out.data_ptr();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    k_relu_fwd<<<blocks, threads>>>(in_ptr, out_ptr, n);
    CHECK(cudaGetLastError());

    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    // Đảm bảo grad_output cũng ở trên GPU
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::empty(input_cache.sizes, DeviceType::CUDA);

    int n = input_cache.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    k_relu_bwd<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)dX.data_ptr(), 
        n
    );
    CHECK(cudaGetLastError());
    return dX;
}