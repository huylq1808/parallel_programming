#pragma once
#include "Tensor.h"

// --- CPU OPS ---

// basic operation - MatMul
void cpu_matmul(const Tensor& A, const Tensor& B, Tensor& C);
void cpu_add(const Tensor& A, const Tensor& B, Tensor& C);

// Conv2D
void cpu_conv2d(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p);
void cpu_conv2d_backward(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                         Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p);

// ReLU
void cpu_relu_forward(const Tensor& in, Tensor& out);
void cpu_relu_backward(const Tensor& in, const Tensor& grad_out, Tensor& grad_in);

// MaxPool2D
void cpu_maxpool2d_forward(const Tensor& in, Tensor& out, Tensor& indices, int k, int s);
void cpu_maxpool2d_backward(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in);

// Upsample2D (Nearest Neighbor)
void cpu_upsample2d_forward(const Tensor& in, Tensor& out, int scale);
void cpu_upsample2d_backward(const Tensor& grad_out, Tensor& grad_in, int scale);

// --- Sigmoid ---
void cpu_sigmoid_forward(const Tensor& in, Tensor& out);
void cpu_sigmoid_backward(const Tensor& out_cache, const Tensor& grad_out, Tensor& grad_in);

// --- CPU OPS: LOSS & OPTIMIZER ---
// Tính MSE Forward (trả về float loss)
float cpu_mse_loss(const Tensor& pred, const Tensor& target);
void cpu_mse_backward(const Tensor& pred, const Tensor& target, Tensor& grad_input);

// SGD Update: param = param - lr * grad
void cpu_sgd_update(Tensor& param, const Tensor& grad, float lr);

// --- CUDA OPS ---
#ifdef USE_CUDA
void cuda_matmul_dispatch(const Tensor& A, const Tensor& B, Tensor& C);
void cuda_add_dispatch(const Tensor& A, const Tensor& B, Tensor& C);

void cuda_conv2d_dispatch(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p);
void cuda_conv2d_backward_dispatch(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                                   Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p);

void cuda_relu_forward_dispatch(const Tensor& in, Tensor& out);
void cuda_relu_backward_dispatch(const Tensor& in, const Tensor& grad_out, Tensor& grad_in);

void cuda_maxpool2d_forward_dispatch(const Tensor& in, Tensor& out, Tensor& indices, int k, int s);
void cuda_maxpool2d_backward_dispatch(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in);

void cuda_upsample2d_forward_dispatch(const Tensor& in, Tensor& out, int scale);
void cuda_upsample2d_backward_dispatch(const Tensor& grad_out, Tensor& grad_in, int scale);

void cuda_sigmoid_forward_dispatch(const Tensor& in, Tensor& out);
void cuda_sigmoid_backward_dispatch(const Tensor& out_cache, const Tensor& grad_out, Tensor& grad_in);

float cuda_mse_loss_dispatch(const Tensor& pred, const Tensor& target);
void cuda_mse_backward_dispatch(const Tensor& pred, const Tensor& target, Tensor& grad_input);

void cuda_sgd_update_dispatch(Tensor& param, const Tensor& grad, float lr);
#endif