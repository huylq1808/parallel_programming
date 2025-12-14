#include "layers/ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    if (out_cache.sizes != input.sizes) {
        out_cache = Tensor::empty(input.sizes);
    }

    size_t n = input.numel();
    const float* i_ptr = (const float*)input.data_ptr();
    float* o_ptr = (float*)out_cache.data_ptr();
    
    for(size_t i=0; i<n; ++i) {
        o_ptr[i] = (i_ptr[i] > 0.0f) ? i_ptr[i] : 0.0f;
    }
    return out_cache;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor dX = Tensor::empty(input_cache.sizes); // Ko cần zeros vì gán trực tiếp

    size_t n = input_cache.numel();
    const float* i_ptr = (const float*)input_cache.data_ptr();
    const float* go_ptr = (const float*)grad_output.data_ptr();
    float* gi_ptr = (float*)dX.data_ptr();
    
    for(size_t i=0; i<n; ++i) {
        gi_ptr[i] = (i_ptr[i] > 0.0f) ? go_ptr[i] : 0.0f;
    }
    return dX;
}