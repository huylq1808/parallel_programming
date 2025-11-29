#pragma once
#include "ILayer.h"

class Dense : public Layer {
public:
    Tensor W, b;
    Tensor input_cache;

    Dense(int in_features, int out_features);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&W, &b}; }
    std::string name() const override { return "Dense"; }
};