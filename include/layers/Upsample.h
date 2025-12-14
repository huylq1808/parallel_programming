#pragma once
#include "ILayer.h"

class Upsample : public Layer {
public:
    int scale_factor;
    std::vector<int64_t> input_shape_cache; // Lưu size input
    Tensor out_cache;

    Upsample(int scale);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Upsample"; }
};