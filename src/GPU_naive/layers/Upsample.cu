#include "layers/Upsample.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

__global__ void k_upsample_fwd(const float* in, float* out, 
                               int N, int C, int H, int W, int H_out, int W_out, int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Nearest neighbor: chia cho scale
    int in_h = oh / scale;
    int in_w = ow / scale;
    int in_idx = n*(C*H*W) + c*(H*W) + in_h*W + in_w;
    
    out[idx] = in[in_idx];
}

__global__ void k_upsample_bwd(const float* grad_out, float* grad_in, 
                               int N, int C, int H, int W, int H_out, int W_out, int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;

    int in_h = oh / scale;
    int in_w = ow / scale;
    int in_idx = n*(C*H*W) + c*(H*W) + in_h*W + in_w;
    
    // Cộng dồn gradient từ ảnh to về pixel gốc
    atomicAdd(&grad_in[in_idx], grad_out[idx]);
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

    int n_out = out.numel();
    k_upsample_fwd<<<(n_out + 255)/256, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)out.data_ptr(),
        N, C, H, W, H_out, W_out, scale_factor
    );
    CHECK(cudaGetLastError());
    return out;
}

Tensor Upsample::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int n_out = grad_out_gpu.numel();
    int N = dX.sizes[0]; int C = dX.sizes[1]; int H = dX.sizes[2]; int W = dX.sizes[3];
    int H_out = grad_out_gpu.sizes[2]; int W_out = grad_out_gpu.sizes[3];

    k_upsample_bwd<<<(n_out + 255)/256, 256>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)dX.data_ptr(),
        N, C, H, W, H_out, W_out, scale_factor
    );
    CHECK(cudaGetLastError());
    return dX;
}