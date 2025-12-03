#include "NaiveOps.cuh"
#include <cuda_runtime.h>

__global__ void k_conv2d_fwd(const float* in, const float* k, const float* b, float* out,
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

    float sum = b[oc];
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
    atomicAdd(&grad_b[oc], go_val);

    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < K_size; ++kh) {
            for (int kw = 0; kw < K_size; ++kw) {
                int in_h = oh * s - p + kh;
                int in_w = ow * s - p + kw;
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int in_idx = n*(C*H*W) + ic*(H*W) + in_h*W + in_w;
                    int k_idx = oc*(C*K_size*K_size) + ic*(K_size*K_size) + kh*K_size + kw;
                    atomicAdd(&grad_k[k_idx], in[in_idx] * go_val);
                    atomicAdd(&grad_in[in_idx], k[k_idx] * go_val);
                }
            }
        }
    }
}

void cuda_launch_conv2d_forward_naive(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p) {
    int total = out.numel();
    k_conv2d_fwd<<<(total + 255)/256, 256>>>(
        (float*)in.data_ptr(), (float*)k.data_ptr(), (float*)b.data_ptr(), (float*)out.data_ptr(),
        in.sizes[0], in.sizes[1], in.sizes[2], in.sizes[3],
        k.sizes[0], k.sizes[2], out.sizes[2], out.sizes[3], s, p
    );
    CHECK(cudaGetLastError());
}

void cuda_launch_conv2d_backward_naive(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                                       Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p) {
    CHECK(cudaMemset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float)));
    CHECK(cudaMemset(grad_k.data_ptr(), 0, grad_k.numel() * sizeof(float)));
    CHECK(cudaMemset(grad_b.data_ptr(), 0, grad_b.numel() * sizeof(float)));
    int total = grad_out.numel();
    k_conv2d_bwd<<<(total + 255)/256, 256>>>(
        (float*)in.data_ptr(), (float*)k.data_ptr(), (float*)grad_out.data_ptr(),
        (float*)grad_in.data_ptr(), (float*)grad_k.data_ptr(), (float*)grad_b.data_ptr(),
        in.sizes[0], in.sizes[1], in.sizes[2], in.sizes[3],
        k.sizes[0], k.sizes[2], grad_out.sizes[2], grad_out.sizes[3], s, p
    );
    CHECK(cudaGetLastError());
}