#pragma once
#include "ILayer.h"

class Sigmoid : public Layer {
public:
    Tensor output_cache; // Lưu OUTPUT thay vì input để tính đạo hàm nhanh hơn

    Sigmoid() = default;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Sigmoid"; }
};