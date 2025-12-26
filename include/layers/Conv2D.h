#pragma once
#include "ILayer.h"

class Conv2D : public Layer {
public:
    int in_c, out_c, k_size, stride, padding;
    Tensor W, b;
    Tensor input_cache;
    Tensor out_cache;

    Conv2D(int in, int out, int k, int s=1, int p=0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&W, &b}; }
    std::string name() const override { return "Conv2D"; }
    void to(DeviceType device) override;
};