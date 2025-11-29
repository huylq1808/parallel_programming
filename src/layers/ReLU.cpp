#include "../../include/layers/ReLU.h"
#include "../../include/core/Ops.h"

Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    Tensor out = Tensor::zeros(input.sizes, input.device);

    if (input.device == DeviceType::CPU) {
        cpu_relu_forward(input, out);
    } 
    #ifdef USE_CUDA
    else {
        cuda_relu_forward_dispatch(input, out);
    }
    #endif
    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor dX = Tensor::zeros(input_cache.sizes, input_cache.device);

    if (input_cache.device == DeviceType::CPU) {
        cpu_relu_backward(input_cache, grad_output, dX);
    } 
    #ifdef USE_CUDA
    else {
        cuda_relu_backward_dispatch(input_cache, grad_output, dX);
    }
    #endif
    return dX;
}