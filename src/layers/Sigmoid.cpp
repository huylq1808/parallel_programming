#include "layers/Sigmoid.h"
#include "core/Ops.h"

Tensor Sigmoid::forward(const Tensor& input) {
    // 1. Tạo Tensor output cùng kích thước với input
    Tensor out = Tensor::zeros(input.sizes, input.device);

    // 2. Dispatch
    if (input.device == DeviceType::CPU) {
        cpu_sigmoid_forward(input, out);
    } 
    #ifdef USE_CUDA
    else {
        cuda_sigmoid_forward_dispatch(input, out);
    }
    #endif

    // 3. Lưu cache để dùng cho backward
    output_cache = out; 
    return out;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // 1. Tạo Tensor gradient input (dL/dx)
    Tensor grad_input = Tensor::zeros(output_cache.sizes, output_cache.device);

    // 2. Dispatch (Lưu ý: truyền output_cache vào để tính đạo hàm)
    if (output_cache.device == DeviceType::CPU) {
        cpu_sigmoid_backward(output_cache, grad_output, grad_input);
    } 
    #ifdef USE_CUDA
    else {
        cuda_sigmoid_backward_dispatch(output_cache, grad_output, grad_input);
    }
    #endif
    
    return grad_input;
}