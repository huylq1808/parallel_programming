#include "layers/MaxPool2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cfloat>

namespace {
__global__ void k_maxpool_fwd_opt(
    const float* in, float* out, float* indices,
    int C, int H, int W, int H_out, int W_out, int k, int s) 
{
    int n = blockIdx.z; // Batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int vol_one_img = C * H_out * W_out;
    if (idx >= vol_one_img) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out;

    int in_offset_batch = n * (C * H * W);
    int out_global_idx = n * vol_one_img + idx;

    int h_start = oh * s;
    int w_start = ow * s;
    float max_val = -FLT_MAX;
    int max_idx = -1;

    const float* in_img = in + in_offset_batch;

    for (int x = 0; x < k; ++x) {
        for (int y = 0; y < k; ++y) {
            int h_in = h_start + x;
            int w_in = w_start + y;
            if (h_in < H && w_in < W) {
                int in_idx_local = c*(H*W) + h_in*W + w_in;
                float val = in_img[in_idx_local];
                if (val > max_val) {
                    max_val = val;
                    // Lưu index Global để backward dễ dùng
                    max_idx = in_offset_batch + in_idx_local; 
                }
            }
        }
    }
    out[out_global_idx] = max_val;
    indices[out_global_idx] = (float)max_idx;
}

__global__ void k_maxpool_bwd_opt(const float* grad_out, const float* indices, float* grad_in, int vol_one_img) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vol_one_img) {
        int global_idx = n * vol_one_img + idx;
        int max_idx = (int)indices[global_idx];
        if (max_idx != -1) {
            atomicAdd(&grad_in[max_idx], grad_out[global_idx]);
        }
    }
}
} // namespace

MaxPool2D::MaxPool2D(int k, int s) : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    Tensor input_gpu = input.to(DeviceType::CUDA);
    input_shape_cache = input_gpu.sizes;

    int N = input_gpu.sizes[0]; int C = input_gpu.sizes[1]; 
    int H = input_gpu.sizes[2]; int W = input_gpu.sizes[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    Tensor out = Tensor::empty({N, C, H_out, W_out}, DeviceType::CUDA);
    indices_cache = Tensor::empty(out.sizes, DeviceType::CUDA);

    int vol_one_img = C * H_out * W_out;
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_maxpool_fwd_opt<<<grid, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)out.data_ptr(), 
        (float*)indices_cache.data_ptr(),
        C, H, W, H_out, W_out, kernel_size, stride
    );
    CHECK(cudaGetLastError());
    //HECK(cudaDeviceSynchronize());
    return out;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int vol_one_img = grad_out_gpu.sizes[1] * grad_out_gpu.sizes[2] * grad_out_gpu.sizes[3];
    int N = grad_out_gpu.sizes[0];
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_maxpool_bwd_opt<<<grid, 256>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (const float*)indices_cache.data_ptr(), 
        (float*)dX.data_ptr(), 
        vol_one_img
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return dX;
}