#include "layers/Upsample.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

__global__ void k_upsample_fwd_opt(const float* in, float* out, 
                               int C, int H, int W, int H_out, int W_out, int scale) 
{
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C * H_out * W_out;
    if (idx >= vol_one_img) return;

    // --- GIẢI MÃ INDEX (FIXED) ---
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out; 

    // Nearest Neighbor Mapping
    int in_h = oh / scale;
    int in_w = ow / scale;
    
    // Offset
    int in_global_idx = n*(C*H*W) + c*(H*W) + in_h*W + in_w;
    int out_global_idx = n*vol_one_img + idx;
    
    out[out_global_idx] = in[in_global_idx];
}

__global__ void k_upsample_bwd_opt(const float* grad_out, float* grad_in, 
                               int C, int H, int W, int H_out, int W_out, int scale) 
{
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C * H_out * W_out;
    if (idx >= vol_one_img) return;

    // --- GIẢI MÃ INDEX (FIXED) ---
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out; 

    int in_h = oh / scale;
    int in_w = ow / scale;
    
    int in_global_idx = n*(C*H*W) + c*(H*W) + in_h*W + in_w;
    int out_global_idx = n*vol_one_img + idx;
    
    atomicAdd(&grad_in[in_global_idx], grad_out[out_global_idx]);
}
} // namespace

Upsample::Upsample(int scale) : scale_factor(scale) {}

Tensor Upsample::forward(const Tensor& input) {
    Tensor input_gpu = input.to(DeviceType::CUDA);
    input_shape_cache = input_gpu.sizes;

    int N = input_gpu.sizes[0]; int C = input_gpu.sizes[1]; 
    int H = input_gpu.sizes[2]; int W = input_gpu.sizes[3];
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;

    Tensor out = Tensor::empty({N, C, H_out, W_out}, DeviceType::CUDA);

    int vol_out = C * H_out * W_out;
    int threads = 256;
    int blocks = (vol_out + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_upsample_fwd_opt<<<grid, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)out.data_ptr(),
        C, H, W, H_out, W_out, scale_factor
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return out;
}

Tensor Upsample::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int N = dX.sizes[0]; int C = dX.sizes[1]; int H = dX.sizes[2]; int W = dX.sizes[3];
    int H_out = grad_out_gpu.sizes[2]; int W_out = grad_out_gpu.sizes[3];

    int vol_out = C * H_out * W_out;
    int threads = 256;
    int blocks = (vol_out + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_upsample_bwd_opt<<<grid, 256>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)dX.data_ptr(),
        C, H, W, H_out, W_out, scale_factor
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return dX;
}