#include "NaiveOps.cuh"
#include <cuda_runtime.h>

// --- RELU ---
__global__ void k_relu_fwd(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}
__global__ void k_relu_bwd(const float* in, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
}

void cuda_launch_relu_forward_naive(const Tensor& in, Tensor& out) {
    k_relu_fwd<<<(in.numel() + 255)/256, 256>>>((float*)in.data_ptr(), (float*)out.data_ptr(), in.numel());
    CHECK(cudaGetLastError());
}
void cuda_launch_relu_backward_naive(const Tensor& in, const Tensor& grad_out, Tensor& grad_in) {
    k_relu_bwd<<<(in.numel() + 255)/256, 256>>>((float*)in.data_ptr(), (float*)grad_out.data_ptr(), (float*)grad_in.data_ptr(), in.numel());
    CHECK(cudaGetLastError());
}

// --- SIGMOID (STABLE) ---
__global__ void k_sigmoid_fwd(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        if (x > 88.0f) out[i] = 1.0f;
        else if (x < -88.0f) out[i] = 0.0f;
        else out[i] = 1.0f / (1.0f + expf(-x));
    }
}
__global__ void k_sigmoid_bwd(const float* y, const float* go, float* gi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gi[i] = go[i] * y[i] * (1.0f - y[i]);
}

void cuda_launch_sigmoid_forward_naive(const Tensor& in, Tensor& out) {
    k_sigmoid_fwd<<<(in.numel()+255)/256, 256>>>((float*)in.data_ptr(), (float*)out.data_ptr(), in.numel());
    CHECK(cudaGetLastError());
}
void cuda_launch_sigmoid_backward_naive(const Tensor& out_cache, const Tensor& grad_out, Tensor& grad_in) {
    k_sigmoid_bwd<<<(out_cache.numel()+255)/256, 256>>>((float*)out_cache.data_ptr(), (float*)grad_out.data_ptr(), (float*)grad_in.data_ptr(), out_cache.numel());
    CHECK(cudaGetLastError());
}