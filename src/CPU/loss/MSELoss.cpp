#include "loss/MSELoss.h"
#include <vector>

float MSELoss::forward(const Tensor& input, const Tensor& target) {
    input_cache = input;
    target_cache = target;

    const float* p_ptr = (const float*)input.data_ptr();
    const float* t_ptr = (const float*)target.data_ptr();
    size_t n = input.numel();
    
    float sum_sq = 0.0f;
    for(size_t i=0; i<n; ++i) {
        float diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
    }
    
    // MSE = Sum((X-Y)^2) / N
    return sum_sq / n;
}

Tensor MSELoss::backward() {
    // dL/dX = 2/N * (X - Y)
    Tensor dInput = Tensor::zeros(input_cache.sizes);
    
    size_t n = input_cache.numel();
    const float* p_ptr = (const float*)input_cache.data_ptr();
    const float* t_ptr = (const float*)target_cache.data_ptr();
    float* g_ptr = (float*)dInput.data_ptr();
    
    float scale = 2.0f / n;
    
    for(size_t i=0; i<n; ++i) {
        g_ptr[i] = scale * (p_ptr[i] - t_ptr[i]);
    }
    
    return dInput;
}