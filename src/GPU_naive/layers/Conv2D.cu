#include "layers/Conv2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>

// ======================================================================
// 1. KERNELS (Internal)
// ======================================================================
namespace {

__global__ void k_conv2d_fwd(const float* in, const float* k, const float* b, float* out,
                             int N, int C, int H, int W, 
                             int Cout, int K_size, int H_out, int W_out, int s, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * H_out * W_out) return;

    // Decode index
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int oc = tmp % Cout;
    int n = tmp / Cout;

    float sum = b[oc]; // Init with bias
    
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K_size; ++kh) {
            for (int kw = 0; kw < K_size; ++kw) {
                int in_h = oh * s - p + kh;
                int in_w = ow * s - p + kw;
                
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int in_idx = n*(C*H*W) + ic*(H*W) + in_h*W + in_w;
                    int k_idx = oc*(C*K_size*K_size) + ic*(K_size*K_size) + kh*K_size + kw;
                    sum += in[in_idx] * k[k_idx];
                }
            }
        }
    }
    out[idx] = sum;
}

__global__ void k_conv2d_bwd(const float* in, const float* k, const float* grad_out,
                             float* grad_in, float* grad_k, float* grad_b,
                             int N, int C, int H, int W, 
                             int Cout, int K_size, int H_out, int W_out, int s, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * H_out * W_out) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int oc = tmp % Cout;
    int n = tmp / Cout;

    float go_val = grad_out[idx];

    // 1. Gradient Bias (Accumulate)
    atomicAdd(&grad_b[oc], go_val);

    // 2. Gradient Weight & Input
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K_size; ++kh) {
            for (int kw = 0; kw < K_size; ++kw) {
                int in_h = oh * s - p + kh;
                int in_w = ow * s - p + kw;
                
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int in_idx = n*(C*H*W) + ic*(H*W) + in_h*W + in_w;
                    int k_idx = oc*(C*K_size*K_size) + ic*(K_size*K_size) + kh*K_size + kw;
                    
                    // dW += Input * Grad_Out
                    atomicAdd(&grad_k[k_idx], in[in_idx] * go_val);
                    // dX += Weight * Grad_Out
                    atomicAdd(&grad_in[in_idx], k[k_idx] * go_val);
                }
            }
        }
    }
}

} // namespace

// ======================================================================
// 2. IMPLEMENTATION
// ======================================================================

Conv2D::Conv2D(int in, int out, int k, int s, int p) 
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p) 
{
    // Init weights... (Giữ nguyên code cũ)
    W = Tensor::randn({out, in, k, k}, 0.0f, 0.08f);
    b = Tensor::zeros({out});
    W.requires_grad = true; b.requires_grad = true;
    
    // Move to GPU immediately
    W = W.to(DeviceType::CUDA);
    b = b.to(DeviceType::CUDA);
}

Tensor Conv2D::forward(const Tensor& input) {
    // 1. Đảm bảo Input ở trên GPU
    input_cache = input.to(DeviceType::CUDA);

    int N = input_cache.sizes[0]; 
    int H = input_cache.sizes[2]; 
    int W_in = input_cache.sizes[3];
    int H_out = (H + 2*padding - k_size) / stride + 1;
    int W_out = (W_in + 2*padding - k_size) / stride + 1;

    // 2. Tạo output tensor mới (Local variable, không dùng out_cache member)
    Tensor out = Tensor::empty({N, out_c, H_out, W_out}, DeviceType::CUDA);

    int total_threads = out.numel();
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    // 3. Gọi Kernel với pointer từ input_cache
    k_conv2d_fwd<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)W.data_ptr(), 
        (const float*)b.data_ptr(), 
        (float*)out.data_ptr(),
        N, in_c, H, W_in, 
        out_c, k_size, H_out, W_out, stride, padding
    );
    CHECK(cudaGetLastError());

    return out;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);

    if(!W.grad) W.ensure_grad();
    if(!b.grad) b.ensure_grad();

    Tensor dIn = Tensor::zeros(input_cache.sizes, DeviceType::CUDA);
    
    // Reset gradients
    W.zero_grad();
    b.zero_grad();

    int total_threads = grad_out_gpu.numel();
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    k_conv2d_bwd<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)W.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(),
        (float*)dIn.data_ptr(), 
        (float*)W.grad->data_ptr(), 
        (float*)b.grad->data_ptr(),
        input_cache.sizes[0], input_cache.sizes[1], input_cache.sizes[2], input_cache.sizes[3],
        out_c, k_size, grad_out_gpu.sizes[2], grad_out_gpu.sizes[3], stride, padding
    );
    CHECK(cudaGetLastError());
    return dIn;
}

void Conv2D::to(DeviceType device){
    W = W.to(device);
    b = b.to(device);
}