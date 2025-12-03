#pragma once
#include "core/Tensor.h"
#include "core/CheckError.h"

// --- ELEMENT-WISE ---
void cuda_launch_add_naive(const Tensor& A, const Tensor& B, Tensor& C);
void cuda_launch_matmul_naive(const Tensor& A, const Tensor& B, Tensor& C);

// --- CONV2D ---
void cuda_launch_conv2d_forward_naive(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p);
void cuda_launch_conv2d_backward_naive(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                                       Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p);

// --- ACTIVATIONS (ReLU + Sigmoid) ---
void cuda_launch_relu_forward_naive(const Tensor& in, Tensor& out);
void cuda_launch_relu_backward_naive(const Tensor& in, const Tensor& grad_out, Tensor& grad_in);

void cuda_launch_sigmoid_forward_naive(const Tensor& in, Tensor& out);
void cuda_launch_sigmoid_backward_naive(const Tensor& out_cache, const Tensor& grad_out, Tensor& grad_in);

// --- POOLING & UPSAMPLE ---
void cuda_launch_maxpool2d_forward_naive(const Tensor& in, Tensor& out, Tensor& indices, int k, int s);
void cuda_launch_maxpool2d_backward_naive(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in);

void cuda_launch_upsample2d_forward_naive(const Tensor& in, Tensor& out, int scale);
void cuda_launch_upsample2d_backward_naive(const Tensor& grad_out, Tensor& grad_in, int scale);

// --- LOSS & OPTIMIZER ---
float cuda_launch_mse_loss_forward_naive(const Tensor& pred, const Tensor& target);
void cuda_launch_mse_loss_backward_naive(const Tensor& pred, const Tensor& target, Tensor& grad_input);
void cuda_launch_sgd_update_naive(Tensor& param, const Tensor& grad, float lr);