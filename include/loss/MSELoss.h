#pragma once
#include "../core/Tensor.h"

class MSELoss {
public:
    Tensor input_cache;
    Tensor target_cache;

    // Forward: Tính ra một số thực (Scalar Loss)
    float forward(const Tensor& input, const Tensor& target);

    // Backward: Tính đạo hàm trả về dInput (dL/dX)
    Tensor backward();
};