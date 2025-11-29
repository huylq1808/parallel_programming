#pragma once
#include <vector>
#include <string>
#include "../core/Tensor.h"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; }
    virtual std::string name() const = 0;
};