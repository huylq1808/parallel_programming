#include "layers/Sigmoid.h"
#include <cmath>

Tensor Sigmoid::forward(const Tensor& input) {
    if (output_cache.sizes != input.sizes) {
        output_cache = Tensor::empty(input.sizes);
    }

    size_t n = input.numel();
    const float* in_ptr = (const float*)input.data_ptr();
    float* out_ptr = (float*)output_cache.data_ptr();

    for(size_t i = 0; i < n; ++i) {
        float x = in_ptr[i];
        // Fix NaN/Overflow
        if (x > 88.0f) {
            out_ptr[i] = 1.0f; 
        } else if (x < -88.0f) {
            out_ptr[i] = 0.0f;
        } else {
            out_ptr[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }
    return output_cache;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor::empty(output_cache.sizes);

    size_t n = output_cache.numel();
    const float* y_ptr = (const float*)output_cache.data_ptr();
    const float* go_ptr = (const float*)grad_output.data_ptr();
    float* gi_ptr = (float*)grad_input.data_ptr();

    for(size_t i = 0; i < n; ++i) {
        float y = y_ptr[i];
        gi_ptr[i] = go_ptr[i] * y * (1.0f - y);
    }
    return grad_input;
}