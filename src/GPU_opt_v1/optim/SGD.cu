#include "optim/SGD.h"
#include "core/Tensor.h"
#include "core/CheckError.h"
#include <cuda_runtime.h>
#include <iostream>

namespace {

__global__ void k_sgd(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}

} // namespace

SGD::SGD(std::vector<Tensor*> params, float lr) 
    : parameters(params), learning_rate(lr) {}

void SGD::zero_grad() {
    for (auto p : parameters) {
        if (p) p->zero_grad();
    }
}

void SGD::step() {
    for (auto p : parameters) {
        if (p && p->grad) {
            // Kiểm tra an toàn: Chỉ chạy kernel nếu tensor nằm trên GPU
            if (p->device != DeviceType::CUDA) {
                std::cerr << "[SGD Warning] Parameter is not on GPU, skipping update!" << std::endl;
                continue;
            }

            int n = p->numel();
            int threads = 256;
            int blocks = (n + threads - 1) / threads;

            k_sgd<<<blocks, threads>>>(
                (float*)p->data_ptr(), 
                (const float*)p->grad->data_ptr(), 
                learning_rate, 
                n
            );
            CHECK(cudaGetLastError());
        }
    }
}