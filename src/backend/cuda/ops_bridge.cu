#pragma once
#include "core/Ops.h"
#include "core/Config.h"
#include "core/CheckError.h"
#include <iostream>


#include "kernels/naive/NaiveOps.cuh"

#ifdef USE_CUDA

// --- Matrix Operations ---
void cuda_matmul_dispatch(const Tensor& A, const Tensor& B, Tensor& C) {
    if (Config::use_optimized_gpu) {
        std::cout << "[GPU] Using OPTIMIZED Matmul\n";
        // Call your Optimized Kernel here
    } else {
        cuda_launch_matmul_naive(A, B, C);
    }
    CHECK(cudaGetLastError());
}

void cuda_add_dispatch(const Tensor& A, const Tensor& B, Tensor& C) {
    cuda_launch_add_naive(A, B, C);
    CHECK(cudaGetLastError());
}

// --- Conv2D Operations ---
void cuda_conv2d_dispatch(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p) {
    cuda_launch_conv2d_forward_naive(in, k, b, out, s, p);
    CHECK(cudaGetLastError());
}

void cuda_conv2d_backward_dispatch(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                                   Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p) {
    cuda_launch_conv2d_backward_naive(in, k, grad_out, grad_in, grad_k, grad_b, s, p);
    CHECK(cudaGetLastError());
}

// --- ReLU Operations ---
void cuda_relu_forward_dispatch(const Tensor& in, Tensor& out) {
    cuda_launch_relu_forward_naive(in, out);
    CHECK(cudaGetLastError());
}

void cuda_relu_backward_dispatch(const Tensor& in, const Tensor& grad_out, Tensor& grad_in) {
    cuda_launch_relu_backward_naive(in, grad_out, grad_in);
    CHECK(cudaGetLastError());
}

// --- MaxPool2D Operations ---
void cuda_maxpool2d_forward_dispatch(const Tensor& in, Tensor& out, Tensor& indices, int k, int s) {
    cuda_launch_maxpool2d_forward_naive(in, out, indices, k, s);
    CHECK(cudaGetLastError());
}

// Lưu ý: Signature phải khớp với Ops.h
void cuda_maxpool2d_backward_dispatch(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in) {
    cuda_launch_maxpool2d_backward_naive(grad_out, indices, grad_in);
    CHECK(cudaGetLastError());
}


// --- Upsample Operations ---
void cuda_upsample2d_forward_dispatch(const Tensor& in, Tensor& out, int scale) {
    cuda_launch_upsample2d_forward_naive(in, out, scale);
    CHECK(cudaGetLastError());
}

void cuda_upsample2d_backward_dispatch(const Tensor& grad_out, Tensor& grad_in, int scale) {
    cuda_launch_upsample2d_backward_naive(grad_out, grad_in, scale);
    CHECK(cudaGetLastError());
}

// --- Sigmoid Operations ---
void cuda_sigmoid_forward_dispatch(const Tensor& in, Tensor& out) {
    cuda_launch_sigmoid_forward_naive(in, out);
    CHECK(cudaGetLastError());
}

void cuda_sigmoid_backward_dispatch(const Tensor& out_cache, const Tensor& grad_out, Tensor& grad_in) {
    cuda_launch_sigmoid_backward_naive(out_cache, grad_out, grad_in);
    CHECK(cudaGetLastError());
}

// --- MSE Loss Dispatchers ---
float cuda_mse_loss_dispatch(const Tensor& pred, const Tensor& target) {
    float loss = cuda_launch_mse_loss_forward_naive(pred, target);
    CHECK(cudaGetLastError());
    return loss;
}

void cuda_mse_backward_dispatch(const Tensor& pred, const Tensor& target, Tensor& grad_input) {
    cuda_launch_mse_loss_backward_naive(pred, target, grad_input);
    CHECK(cudaGetLastError());
}

// --- SGD Dispatcher ---
void cuda_sgd_update_dispatch(Tensor& param, const Tensor& grad, float lr) {
    cuda_launch_sgd_update_naive(param, grad, lr);
    CHECK(cudaGetLastError());
    // Gọi kernel element-wise: w = w - lr * g
}

#endif