#pragma once
#include "ILayer.h"

class ReLU : public Layer {
public:
    Tensor input_cache; // Lưu input để tính đạo hàm
    Tensor out_cache;

    ReLU() = default;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "ReLU"; }
};