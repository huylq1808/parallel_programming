#include "NaiveOps.cuh"
#include <cuda_runtime.h>

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
    atomicAdd(&grad_in[in_idx], grad_out[idx]);
}

void cuda_launch_upsample2d_forward_naive(const Tensor& in, Tensor& out, int scale) {
    k_upsample_fwd<<<(out.numel() + 255)/256, 256>>>(
        (float*)in.data_ptr(), (float*)out.data_ptr(),
        in.sizes[0], in.sizes[1], in.sizes[2], in.sizes[3],
        out.sizes[2], out.sizes[3], scale
    );
    CHECK(cudaGetLastError());
}

void cuda_launch_upsample2d_backward_naive(const Tensor& grad_out, Tensor& grad_in, int scale) {
    CHECK(cudaMemset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float)));
    k_upsample_bwd<<<(grad_out.numel() + 255)/256, 256>>>(
        (float*)grad_out.data_ptr(), (float*)grad_in.data_ptr(),
        grad_in.sizes[0], grad_in.sizes[1], grad_in.sizes[2], grad_in.sizes[3],
        grad_out.sizes[2], grad_out.sizes[3], scale
    );
    CHECK(cudaGetLastError());
}