#include "optim/SGD.h"

SGD::SGD(std::vector<Tensor*> params, float lr) 
    : parameters(params), learning_rate(lr) {}

void SGD::zero_grad() {
    for (auto p : parameters) {
        if (p) p->zero_grad();
    }
}

void SGD::step() {
    for (auto p : parameters) {
        // Chỉ update nếu tensor có gradient
        if (p && p->grad) {
            size_t n = p->numel();
            float* w_ptr = (float*)p->data_ptr();
            const float* g_ptr = (const float*)p->grad->data_ptr();
            
            for(size_t i=0; i<n; ++i) {
                w_ptr[i] -= learning_rate * g_ptr[i];
            }
        }
    }
}