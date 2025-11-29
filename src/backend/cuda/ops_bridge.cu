#include "../../../include/core/Ops.h"
#include "../../../include/core/Config.h"
#include <iostream>

#ifdef USE_CUDA

// --- Matrix Operations ---
void cuda_matmul_dispatch(const Tensor& A, const Tensor& B, Tensor& C) {
    if (Config::use_optimized_gpu) {
        std::cout << "[GPU] Using OPTIMIZED Matmul\n";
        // Call your Optimized Kernel here
    } else {
        std::cout << "[GPU] Using NAIVE Matmul\n";
        // Call your Naive Kernel here
    }
}
// --- Conv2D Operations ---
void cuda_conv2d_dispatch(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int s, int p) {
    std::cout << "[GPU] Conv2D Forward called (Placeholder)\n";
}

void cuda_conv2d_backward_dispatch(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                                   Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int s, int p) {
    std::cout << "[GPU] Conv2D Backward called (Placeholder)\n";
}

// --- ReLU Operations ---
void cuda_relu_forward_dispatch(const Tensor& in, Tensor& out) {
    std::cout << "[GPU] ReLU Forward called (Placeholder)\n";
}

void cuda_relu_backward_dispatch(const Tensor& in, const Tensor& grad_out, Tensor& grad_in) {
    std::cout << "[GPU] ReLU Backward called (Placeholder)\n";
}

// --- MaxPool2D Operations ---
void cuda_maxpool2d_forward_dispatch(const Tensor& in, Tensor& out, Tensor& indices, int k, int s) {
    std::cout << "[GPU] MaxPool2D Forward called (Placeholder)\n";
}

// Lưu ý: Signature phải khớp với Ops.h
void cuda_maxpool2d_backward_dispatch(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in) {
    std::cout << "[GPU] MaxPool2D Backward called (Placeholder)\n";
}

// --- Upsample Operations ---
void cuda_upsample2d_forward_dispatch(const Tensor& in, Tensor& out, int scale) {
    std::cout << "[GPU] Upsample2D Forward called (Placeholder)\n";
}

void cuda_upsample2d_backward_dispatch(const Tensor& grad_out, Tensor& grad_in, int scale) {
    std::cout << "[GPU] Upsample2D Backward called (Placeholder)\n";
}

// --- MSE Loss Dispatchers ---
float cuda_mse_loss_dispatch(const Tensor& pred, const Tensor& target) {
    std::cout << "[GPU] MSE Forward (Placeholder - Returning 0.0)\n";
    // Lưu ý: Forward loss trên GPU cần kernel Reduction (tính tổng song song).
    // Sau khi tính xong trên GPU, cần copy 1 giá trị float về CPU để trả về.
    return 0.0f; 
}

void cuda_mse_backward_dispatch(const Tensor& pred, const Tensor& target, Tensor& grad_input) {
    std::cout << "[GPU] MSE Backward (Placeholder)\n";
    // Gọi kernel element-wise: grad = 2/N * (pred - target)
}

// --- SGD Dispatcher ---
void cuda_sgd_update_dispatch(Tensor& param, const Tensor& grad, float lr) {
    std::cout << "[GPU] SGD Update (Placeholder)\n";
    // Gọi kernel element-wise: w = w - lr * g
}

#endif