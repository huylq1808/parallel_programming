#include <iostream>
#include <string>
#include "../include/core/Tensor.h"
#include "../include/layers/Conv2D.h"
#include "../include/core/Config.h"

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "cpu";
    std::cout << ">>> RUNNING MODE: " << mode << std::endl;

    DeviceType dev = DeviceType::CPU;
    if (mode == "gpu_naive") { dev = DeviceType::CUDA; Config::use_optimized_gpu = false; }
    if (mode == "gpu_opt")   { dev = DeviceType::CUDA; Config::use_optimized_gpu = true; }

    // 1. Data Setup (Fake CIFAR-10)
    Tensor x = Tensor::randn({2, 3, 32, 32}, 0, 1, dev);
    
    // 2. Model Setup
    Conv2D conv(3, 16, 3, 1, 1);
    // Nếu chạy GPU thì cần chuyển trọng số sang GPU
    if (dev == DeviceType::CUDA) {
        conv.W = conv.W.to(DeviceType::CUDA);
        conv.b = conv.b.to(DeviceType::CUDA);
    }

    // 3. Forward
    std::cout << "Forwarding..." << std::endl;
    Tensor y = conv.forward(x);
    std::cout << "Output shape: " << y.sizes[1] << "x" << y.sizes[2] << "x" << y.sizes[3] << std::endl;

    // 4. Backward
    std::cout << "Backwarding..." << std::endl;
    Tensor dy = Tensor::randn(y.sizes, 0, 1, dev); 
    conv.backward(dy);

    std::cout << "Done!" << std::endl;
    return 0;
}