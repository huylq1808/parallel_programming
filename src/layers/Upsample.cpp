#include "../../include/layers/Upsample.h"
#include "../../include/core/Ops.h"

Upsample::Upsample(int scale) : scale_factor(scale) {}

Tensor Upsample::forward(const Tensor& input) {
    input_shape_cache = input.sizes; // Lưu shape để backward biết size dX
    int N = input.sizes[0]; int C = input.sizes[1]; int H = input.sizes[2]; int W = input.sizes[3];
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;

    Tensor out = Tensor::zeros({N, C, H_out, W_out}, input.device);

    if (input.device == DeviceType::CPU) {
        cpu_upsample2d_forward(input, out, scale_factor);
    } 
    #ifdef USE_CUDA
    else {
        cuda_upsample2d_forward_dispatch(input, out, scale_factor);
    }
    #endif
    return out;
}

Tensor Upsample::backward(const Tensor& grad_output) {
    Tensor dX = Tensor::zeros(input_shape_cache, grad_output.device);

    if (grad_output.device == DeviceType::CPU) {
        cpu_upsample2d_backward(grad_output, dX, scale_factor);
    } 
    #ifdef USE_CUDA
    else {
        cuda_upsample2d_backward_dispatch(grad_output, dX, scale_factor);
    }
    #endif
    return dX;
}