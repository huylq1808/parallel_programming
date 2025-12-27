#include "layers/Conv2D_relu.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace {

//FORWARD (Fused) 
__global__ void k_conv2d_relu_fwd_opt(
    const float* __restrict__ in, 
    const float* __restrict__ k, 
    const float* __restrict__ b, 
    float* __restrict__ out,
    int C_in, int H_in, int W_in, 
    int C_out, int H_out, int W_out, 
    int K_size, int stride, int padding) 
{
    int n = blockIdx.z; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C_out * H_out * W_out;

    if (idx >= vol_one_img) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int oc = tmp / H_out; 

    int in_offset_batch = n * (C_in * H_in * W_in);
    int out_global_idx = n * vol_one_img + idx;

    float sum = b[oc]; 
    const float* in_img = in + in_offset_batch;

    for (int ic = 0; ic < C_in; ++ic) {
        int in_offset_c = ic * H_in * W_in;
        int k_offset_c  = oc * (C_in * K_size * K_size) + ic * (K_size * K_size);
        
        #pragma unroll
        for (int kh = 0; kh < K_size; ++kh) {
          #pragma unroll
            for (int kw = 0; kw < K_size; ++kw) {
                int h_in = oh * stride - padding + kh;
                int w_in = ow * stride - padding + kw;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    sum += in_img[in_offset_c + h_in * W_in + w_in] * k[k_offset_c + kh * K_size + kw];
                }
            }
        }
    }
    // ReLU
    out[out_global_idx] = (sum > 0.0f) ? sum : 0.0f;
}

//  OPTIMIZED BIAS BACKWARD
__global__ void k_conv2d_relu_bwd_bias(
    const float* __restrict__ grad_out, 
    const float* __restrict__ fwd_out, // Needed for ReLU mask
    float* __restrict__ grad_b, 
    int N, int C_out, int H_out, int W_out) 
{
    int oc = blockIdx.x; 
    if (oc >= C_out) return;

    int vol_channel = H_out * W_out;
    int total_elements = N * vol_channel; 

    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int n = i / vol_channel;
        int rem = i % vol_channel;
        int idx = n * (C_out * vol_channel) + oc * vol_channel + rem;
        
        // Only accumulate if forward output > 0
        if (fwd_out[idx] > 0.0f) {
            local_sum += grad_out[idx];
        }
    }

    // Warp Reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&grad_b[oc], local_sum);
    }
}

//OPTIMIZED DATA/WEIGHT BACKWARD
__global__ void k_conv2d_relu_bwd_data_weights(
    const float* __restrict__ in, 
    const float* __restrict__ k, 
    const float* __restrict__ grad_out,
    const float* __restrict__ fwd_out,
    float* __restrict__ grad_in, 
    float* __restrict__ grad_k, 
    int C_in, int H_in, int W_in, 
    int C_out, int H_out, int W_out, 
    int K_size, int stride, int padding) 
{
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C_out * H_out * W_out;

    if (idx >= vol_one_img) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int oc = tmp / H_out; 

    int in_offset_batch = n * (C_in * H_in * W_in);
    int global_idx = n * vol_one_img + idx;

    float go_val = grad_out[global_idx];

    // FUSED RELU CHECK
    if (fwd_out[global_idx] <= 0.0f) {
        return; // Gradient killed by ReLU
    }
    // Additional optimization for sparsity
    if (abs(go_val) < 1e-9) return;

    const float* in_img = in + in_offset_batch;
    float* gi_img = grad_in + in_offset_batch;

    for (int ic = 0; ic < C_in; ++ic) {
        int in_offset_c = ic * H_in * W_in;
        int k_offset_c  = oc * (C_in * K_size * K_size) + ic * (K_size * K_size);

        #pragma unroll
        for (int kh = 0; kh < K_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < K_size; ++kw) {
                int h_in = oh * stride - padding + kh;
                int w_in = ow * stride - padding + kw;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int in_idx = in_offset_c + h_in * W_in + w_in;
                    int k_idx  = k_offset_c + kh * K_size + kw;
                    
                    atomicAdd(&grad_k[k_idx], in_img[in_idx] * go_val);
                    atomicAdd(&gi_img[in_idx], k[k_idx] * go_val);
                }
            }
        }
    }
}
} // namespace


Conv2D_relu::Conv2D_relu(int in, int out, int k, int s, int p) 
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p) 
{
    W = Tensor::randn({out, in, k, k}, 0.0f, 0.08f, DeviceType::CUDA);
    b = Tensor::zeros({out}, DeviceType::CUDA);
    W.requires_grad = true; b.requires_grad = true;
}

Tensor Conv2D_relu::forward(const Tensor& input) {
    input_cache = input.to(DeviceType::CUDA);
    int N = input_cache.sizes[0];
    int H = input_cache.sizes[2]; 
    int W_in = input_cache.sizes[3];
    int H_out = (H + 2*padding - k_size) / stride + 1;
    int W_out = (W_in + 2*padding - k_size) / stride + 1;

    out_cache = Tensor::empty({N, out_c, H_out, W_out}, DeviceType::CUDA);

    int vol_one_img = out_c * H_out * W_out;
    int threads = 256;
    int blocks_x = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks_x, 1, N); 
    dim3 block(threads, 1, 1);

    k_conv2d_relu_fwd_opt<<<grid, block>>>(
        (const float*)input_cache.data_ptr(), (const float*)W.data_ptr(), (const float*)b.data_ptr(), (float*)out_cache.data_ptr(),
        in_c, H, W_in, out_c, H_out, W_out, k_size, stride, padding
    );
    CHECK(cudaGetLastError());
    return out_cache;
}

Tensor Conv2D_relu::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    if(!W.grad) W.ensure_grad();
    if(!b.grad) b.ensure_grad();
    
    Tensor dIn = Tensor::zeros(input_cache.sizes, DeviceType::CUDA);
    W.zero_grad();
    b.zero_grad();

    int N = input_cache.sizes[0];
    int H = input_cache.sizes[2];
    int W_in = input_cache.sizes[3];
    int H_out = grad_out_gpu.sizes[2];
    int W_out = grad_out_gpu.sizes[3];

    // 1. Bias Backward (With ReLU logic)
    int threads_bias = 256;
    k_conv2d_relu_bwd_bias<<<out_c, threads_bias>>>(
        (const float*)grad_out_gpu.data_ptr(),
        (const float*)out_cache.data_ptr(), // Fwd output for mask
        (float*)b.grad->data_ptr(),
        N, out_c, H_out, W_out
    );

    // 2. Data/Weights Backward
    int vol_one_img = out_c * H_out * W_out;
    int threads = 256;
    int blocks_x = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks_x, 1, N); 
    dim3 block(threads, 1, 1);

    k_conv2d_relu_bwd_data_weights<<<grid, block>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)W.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(),
        (const float*)out_cache.data_ptr(),
        (float*)dIn.data_ptr(), 
        (float*)W.grad->data_ptr(), 
        in_c, H, W_in, out_c, H_out, W_out, k_size, stride, padding
    );
    CHECK(cudaGetLastError());
    return dIn;
}

void Conv2D_relu::to(DeviceType device){
    W = W.to(device);
    b = b.to(device);
}