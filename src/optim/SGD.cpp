#include "../../include/optim/SGD.h"
#include "../../include/core/Ops.h"

SGD::SGD(std::vector<Tensor*> params, float lr) 
    : parameters(params), learning_rate(lr) {}

void SGD::zero_grad() {
    for (auto p : parameters) {
        if (p) p->zero_grad();
    }
}

void SGD::step() {
    for (auto p : parameters) {
        // Chỉ update nếu tensor này có gradient (tức là requires_grad=true)
        if (p && p->grad) {
            if (p->device == DeviceType::CPU) {
                cpu_sgd_update(*p, *p->grad, learning_rate);
            }
            #ifdef USE_CUDA
            else {
                cuda_sgd_update_dispatch(*p, *p->grad, learning_rate);
            }
            #endif
        }
    }
}