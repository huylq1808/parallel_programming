#include "loss/MSELoss.h"
#include "core/Ops.h"
#include <iostream>

// this version now is for CPU only, need to add CUDA version later

float MSELoss::forward(const Tensor& input, const Tensor& target) {
    input_cache = input;
    target_cache = target;

    if (input.device == DeviceType::CPU) {
        return cpu_mse_loss(input, target); // Cần đảm bảo hàm này trả về float
    }
    #ifdef USE_CUDA
    else {
        // Hàm này cần launch kernel, copy kết quả loss từ GPU về CPU và trả về float
        return cuda_mse_loss_dispatch(input, target); 
    }
    #endif
    return 0.0f;
}

Tensor MSELoss::backward() {
    //  MSE: dL/dX = 2/N * (X - Y)
    Tensor dInput = Tensor::zeros(input_cache.sizes, input_cache.device);
    if (input_cache.device == DeviceType::CPU) {
        cpu_mse_backward(input_cache, target_cache, dInput);
    }
    #ifdef USE_CUDA
    else {
        cuda_mse_backward_dispatch(input_cache, target_cache, dInput);
    }
    #endif
    return dInput;
}