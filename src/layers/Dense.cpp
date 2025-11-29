#include "../../include/layers/Dense.h"

Dense::Dense(int in_f, int out_f) {
    W = Tensor::randn({in_f, out_f}, 0.0f, 0.1f);
    b = Tensor::zeros({out_f});
    W.requires_grad = true; b.requires_grad = true;
}

Tensor Dense::forward(const Tensor& input) {
    input_cache = input;
    return input.matmul(W).add(b);
}

Tensor Dense::backward(const Tensor& grad_output) {
    if(!W.grad) W.ensure_grad();
    if(!b.grad) b.ensure_grad();

    // dW = Input^T * Grad
    *(W.grad) = input_cache.transpose(0, 1).matmul(grad_output);
    
    // dX = Grad * W^T
    Tensor dX = grad_output.matmul(W.transpose(0, 1));

    // db = Sum(Grad, axis=0) -> Placeholder logic for now
    
    
    return dX;
}