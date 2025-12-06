#include "layers/ReLU.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {
__global__ void k_relu_fwd_opt(const float* in, float* out, int vol_one_img) {
    int n = blockIdx.z; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vol_one_img) {
        int global_idx = n * vol_one_img + idx;
        float val = in[global_idx];
        out[global_idx] = (val > 0.0f) ? val : 0.0f;
    }
}

__global__ void k_relu_bwd_opt(const float* in, const float* grad_out, float* grad_in, int vol_one_img) {
    int n = blockIdx.z; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vol_one_img) {
        int global_idx = n * vol_one_img + idx;
        grad_in[global_idx] = (in[global_idx] > 0.0f) ? grad_out[global_idx] : 0.0f;
    }
}
} // namespace

Tensor ReLU::forward(const Tensor& input) {
    input_cache = input.to(DeviceType::CUDA);
    Tensor out = Tensor::empty(input_cache.sizes, DeviceType::CUDA);

    int N = input_cache.sizes[0];
    int vol_one_img = input_cache.numel() / N;
    
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_relu_fwd_opt<<<grid, threads>>>(
        (const float*)input_cache.data_ptr(), (float*)out.data_ptr(), vol_one_img);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::empty(input_cache.sizes, DeviceType::CUDA);

    int N = input_cache.sizes[0];
    int vol_one_img = input_cache.numel() / N;
    
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_relu_bwd_opt<<<grid, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)dX.data_ptr(), vol_one_img);
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return dX;
}