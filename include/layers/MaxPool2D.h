#pragma once
#include "ILayer.h"

class MaxPool2D : public Layer {
public:
    int kernel_size;    
    int stride;
    Tensor indices_cache; // Quan trọng: Lưu vị trí max

    MaxPool2D(int k, int s);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "MaxPool2D"; }
};