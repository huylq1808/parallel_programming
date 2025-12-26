#include "layers/MaxPool2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cfloat> // FLT_MAX

namespace {

__global__ void k_maxpool_fwd(const float* in, float* out, float* indices,
                               int N, int C, int H, int W, int H_out, int W_out, int k, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = oh * s;
    int w_start = ow * s;
    
    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int x = 0; x < k; ++x) {
        for (int y = 0; y < k; ++y) {
            int h_in = h_start + x;
            int w_in = w_start + y;
            if (h_in < H && w_in < W) {
                int in_idx = n*(C*H*W) + c*(H*W) + h_in*W + w_in;
                if (in[in_idx] > max_val) {
                    max_val = in[in_idx];
                    max_idx = in_idx;
                }
            }
        }
    }
    out[idx] = max_val;
    indices[idx] = (float)max_idx;
}

__global__ void k_maxpool_bwd(const float* grad_out, const float* indices, float* grad_in, int n_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        int max_idx = (int)indices[i];
        if (max_idx != -1) {
            atomicAdd(&grad_in[max_idx], grad_out[i]);
        }
    }
}

} // namespace

MaxPool2D::MaxPool2D(int k, int s) : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    // Chuyển input sang GPU
    Tensor input_gpu = input.to(DeviceType::CUDA);
    input_shape_cache = input_gpu.sizes; // Lưu shape để backward dùng

    int N = input_gpu.sizes[0]; int C = input_gpu.sizes[1]; 
    int H = input_gpu.sizes[2]; int W = input_gpu.sizes[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    // Tạo Output và Indices mới
    Tensor out = Tensor::empty({N, C, H_out, W_out}, DeviceType::CUDA);
    indices_cache = Tensor::empty(out.sizes, DeviceType::CUDA); // indices_cache cần lưu lại cho backward

    int n_out = out.numel();
    k_maxpool_fwd<<<(n_out + 255)/256, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)out.data_ptr(), 
        (float*)indices_cache.data_ptr(),
        N, C, H, W, H_out, W_out, kernel_size, stride
    );
    CHECK(cudaGetLastError());
    return out;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int n_out = grad_out_gpu.numel();
    k_maxpool_bwd<<<(n_out + 255)/256, 256>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (const float*)indices_cache.data_ptr(), 
        (float*)dX.data_ptr(), 
        n_out
    );
    CHECK(cudaGetLastError());
    return dX;
}