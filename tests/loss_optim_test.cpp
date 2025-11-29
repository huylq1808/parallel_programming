#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../include/core/Tensor.h"
#include "../include/loss/MSELoss.h"
#include "../include/optim/SGD.h"

// --- Utils ---
void print_val(std::string name, float val) {
    std::cout << "  " << name << ": " << val << std::endl;
}

void print_tensor_tiny(std::string name, const Tensor& t) {
    std::cout << "  " << name << ": [ ";
    const float* p = (const float*)t.data_ptr();
    for(size_t i=0; i<t.numel(); ++i) std::cout << p[i] << " ";
    std::cout << "]" << std::endl;
}

// --- Test Case 1: MSE Loss Logic ---
void test_mse_logic() {
    std::cout << "\n=== TEST 1: MSE LOSS LOGIC ===" << std::endl;
    
    // Input: [2, 5, 10]
    // Target: [1, 5, 12]
    // Diff:   [1, 0, -2]
    // Sq:     [1, 0, 4]
    // Sum:    5
    // Mean (N=3): 5/3 = 1.666...
    
    Tensor input = Tensor::zeros({3}, DeviceType::CPU);
    float* p_in = (float*)input.data_ptr();
    p_in[0]=2.0; p_in[1]=5.0; p_in[2]=10.0;

    Tensor target = Tensor::zeros({3}, DeviceType::CPU);
    float* p_tar = (float*)target.data_ptr();
    p_tar[0]=1.0; p_tar[1]=5.0; p_tar[2]=12.0;

    MSELoss criterion;
    float loss = criterion.forward(input, target);
    
    print_tensor_tiny("Input", input);
    print_tensor_tiny("Target", target);
    print_val("Calculated Loss", loss);
    print_val("Expected Loss", 5.0f/3.0f);

    // Test Backward
    // Grad = 2/N * (Input - Target)
    // N = 3
    // g[0] = 2/3 * (1) = 0.666
    // g[1] = 2/3 * (0) = 0
    // g[2] = 2/3 * (-2) = -1.333
    Tensor grad = criterion.backward();
    print_tensor_tiny("Gradient (dL/dX)", grad);
}

// --- Test Case 2: SGD Update ---
void test_sgd_update() {
    std::cout << "\n=== TEST 2: SGD OPTIMIZER UPDATE ===" << std::endl;
    
    // Weight = [10.0, -5.0]
    // Grad   = [1.0,  -2.0]  (Giả sử đã tính được từ backward)
    // LR     = 0.1
    
    // New Weight = 10 - 0.1*1 = 9.9
    // New Weight = -5 - 0.1*(-2) = -5 + 0.2 = -4.8

    Tensor param = Tensor::zeros({2}, DeviceType::CPU);
    float* w = (float*)param.data_ptr();
    w[0] = 10.0f; w[1] = -5.0f;
    param.requires_grad = true;

    // Giả lập gradient
    param.ensure_grad();
    float* g = (float*)param.grad->data_ptr();
    g[0] = 1.0f; g[1] = -2.0f;

    print_tensor_tiny("Weight Before", param);
    print_tensor_tiny("Gradient", *param.grad);

    // Init Optimizer
    std::vector<Tensor*> params = {&param};
    SGD optimizer(params, 0.1f); // LR = 0.1

    optimizer.step();
    print_tensor_tiny("Weight After Step", param);

    // Check Zero Grad
    optimizer.zero_grad();
    print_tensor_tiny("Gradient After Zero", *param.grad); // Should be all 0
}

int main() {
    test_mse_logic();
    test_sgd_update();
    return 0;
}