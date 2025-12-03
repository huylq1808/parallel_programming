#include "NaiveOps.cuh"
#include <cuda_runtime.h>
#include <cfloat>

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
        if (max_idx != -1) atomicAdd(&grad_in[max_idx], grad_out[i]);
    }
}

void cuda_launch_maxpool2d_forward_naive(const Tensor& in, Tensor& out, Tensor& indices, int k, int s) {
    k_maxpool_fwd<<<(out.numel() + 255)/256, 256>>>(
        (float*)in.data_ptr(), (float*)out.data_ptr(), (float*)indices.data_ptr(),
        in.sizes[0], in.sizes[1], in.sizes[2], in.sizes[3],
        out.sizes[2], out.sizes[3], k, s
    );
    CHECK(cudaGetLastError());
}
void cuda_launch_maxpool2d_backward_naive(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in) {
    CHECK(cudaMemset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float)));
    k_maxpool_bwd<<<(grad_out.numel() + 255)/256, 256>>>(
        (float*)grad_out.data_ptr(), (float*)indices.data_ptr(), (float*)grad_in.data_ptr(), grad_out.numel()
    );
    CHECK(cudaGetLastError());
}