#pragma once
#include <vector>
#include "../core/Tensor.h"

class SGD {
public:
    std::vector<Tensor*> parameters; // Danh sách các con trỏ trỏ tới Weight/Bias
    float learning_rate;

    SGD(std::vector<Tensor*> params, float lr = 0.01f);

    void zero_grad();
    void step();
};