#include "../../include/layers/Conv2D.h"
#include "../../include/core/Ops.h"

Conv2D::Conv2D(int in, int out, int k, int s, int p) 
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p) 
{
    W = Tensor::randn({out, in, k, k}, 0.0f, 0.1f);
    b = Tensor::zeros({out});
    W.requires_grad = true; b.requires_grad = true;
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;
    int N = input.sizes[0]; int H = input.sizes[2]; int W_in = input.sizes[3];
    int H_out = (H + 2*padding - k_size) / stride + 1;
    int W_out = (W_in + 2*padding - k_size) / stride + 1;
    Tensor out = Tensor::zeros({N, out_c, H_out, W_out}, input.device);

    if (input.device == DeviceType::CPU) {
        cpu_conv2d(input, W, b, out, stride, padding);
    } 
    #ifdef USE_CUDA
    else {
        cuda_conv2d_dispatch(input, W, b, out, stride, padding);
    }
    #endif
    return out;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    if(!W.grad) W.ensure_grad();
    if(!b.grad) b.ensure_grad();
    Tensor dIn = Tensor::zeros(input_cache.sizes, input_cache.device);

    if (input_cache.device == DeviceType::CPU) {
        cpu_conv2d_backward(input_cache, W, grad_output, dIn, *W.grad, *b.grad, stride, padding);
    } 
    #ifdef USE_CUDA
    else {
        cuda_conv2d_backward_dispatch(input_cache, W, grad_output, dIn, *W.grad, *b.grad, stride, padding);
    }
    #endif
    return dIn;
}