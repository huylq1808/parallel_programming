#include "../../include/loss/MSELoss.h"
#include <iostream>

// this version now is for CPU only, need to add CUDA version later

float MSELoss::forward(const Tensor& input, const Tensor& target) {
    input_cache = input;
    target_cache = target;

    size_t n = input.numel();
    const float* in_ptr = (const float*)input.data_ptr();
    const float* tar_ptr = (const float*)target.data_ptr();

    float sum_sq_error = 0.0f;
    
    // TODO: must be re-implememt with parallel programming
    for(size_t i = 0; i < n; ++i) {
        float diff = in_ptr[i] - tar_ptr[i];
        sum_sq_error += diff * diff;
    }

    return sum_sq_error / n;
}

Tensor MSELoss::backward() {
    //  MSE: dL/dX = 2/N * (X - Y)
    size_t n = input_cache.numel();
    Tensor grad = Tensor::zeros(input_cache.sizes, input_cache.device);
    
    const float* in_ptr = (const float*)input_cache.data_ptr();
    const float* tar_ptr = (const float*)target_cache.data_ptr();
    float* g_ptr = (float*)grad.data_ptr();
    
    float scale = 2.0f / n;

    for(size_t i = 0; i < n; ++i) {
        g_ptr[i] = scale * (in_ptr[i] - tar_ptr[i]);
    }

    return grad;
}