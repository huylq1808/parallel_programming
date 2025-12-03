#include "NaiveOps.cuh"
#include <cuda_runtime.h>

__global__ void k_mse_loss(const float* pred, const float* target, float* shared_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = pred[i] - target[i];
        atomicAdd(shared_sum, diff * diff);
    }
}

float cuda_launch_mse_loss_forward_naive(const Tensor& pred, const Tensor& target) {
    int n = pred.numel();
    float* d_sum;
    CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    k_mse_loss<<<(n+255)/256, 256>>>((float*)pred.data_ptr(), (float*)target.data_ptr(), d_sum, n);
    CHECK(cudaGetLastError());

    float h_sum;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sum));
    return h_sum / n;
}

__global__ void k_mse_bwd(const float* pred, const float* target, float* grad, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = scale * (pred[i] - target[i]);
}

void cuda_launch_mse_loss_backward_naive(const Tensor& pred, const Tensor& target, Tensor& grad_input) {
    int n = pred.numel();
    float scale = 2.0f / n;
    k_mse_bwd<<<(n+255)/256, 256>>>((float*)pred.data_ptr(), (float*)target.data_ptr(), (float*)grad_input.data_ptr(), n, scale);
    CHECK(cudaGetLastError());
}

__global__ void k_sgd(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

void cuda_launch_sgd_update_naive(Tensor& param, const Tensor& grad, float lr) {
    k_sgd<<<(param.numel()+255)/256, 256>>>((float*)param.data_ptr(), (float*)grad.data_ptr(), lr, param.numel());
    CHECK(cudaGetLastError());
}