#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

// Include Core
#include "../include/core/Tensor.h"

// Include Layers
#include "../include/layers/Dense.h"
#include "../include/layers/Conv2D.h"
#include "../include/layers/ReLU.h"
#include "../include/layers/MaxPool2D.h"
#include "../include/layers/Upsample.h"

// ======================================================================
// UTILS: HÀM IN TENSOR ĐẸP (HIỂN THỊ DATA)
// ======================================================================

void print_tensor_box(const std::string& name, const Tensor& t) {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "VAR: " << name << " | Shape: [";
    for (size_t i = 0; i < t.sizes.size(); ++i) std::cout << t.sizes[i] << (i < t.sizes.size()-1 ? "," : "");
    std::cout << "]" << std::endl;

    const float* ptr = (const float*)t.data_ptr();
    
    // Logic in đẹp dựa trên số chiều
    if (t.sizes.size() == 1) {
        std::cout << "[ ";
        for(int i=0; i<t.sizes[0]; ++i) std::cout << std::fixed << std::setprecision(1) << ptr[i] << " ";
        std::cout << "]" << std::endl;
    }
    else if (t.sizes.size() == 2) {
        int rows = t.sizes[0];
        int cols = t.sizes[1];
        for(int i=0; i<rows; ++i) {
            std::cout << (i==0 ? "[[ " : " [ ");
            for(int j=0; j<cols; ++j) {
                std::cout << std::setw(5) << std::fixed << std::setprecision(1) << ptr[i*cols + j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }
    else if (t.sizes.size() == 3) {
        int C = t.sizes[0]; int H = t.sizes[1]; int W = t.sizes[2];
        for(int c=0; c<C; ++c) {
            std::cout << "Channel " << c << ":" << std::endl;
            for(int h=0; h<H; ++h) {
                std::cout << "  ";
                for(int w=0; w<W; ++w) {
                    int idx = c*H*W + h*W + w;
                    std::cout << std::setw(5) << std::fixed << std::setprecision(1) << ptr[idx] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    else if (t.sizes.size() == 4) {
        // N, C, H, W (Chỉ in batch đầu tiên để đỡ rối)
        int N = t.sizes[0]; int C = t.sizes[1]; int H = t.sizes[2]; int W = t.sizes[3];
        std::cout << "(Showing Batch 0 only)" << std::endl;
        for(int c=0; c<C; ++c) {
            std::cout << "Batch 0, Channel " << c << ":" << std::endl;
            for(int h=0; h<H; ++h) {
                std::cout << "  ";
                for(int w=0; w<W; ++w) {
                    int idx = 0*C*H*W + c*H*W + h*W + w;
                    std::cout << std::setw(5) << std::fixed << std::setprecision(1) << ptr[idx] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "--------------------------------------------------------\n" << std::endl;
}

// Hàm điền dữ liệu tuần tự 0, 1, 2... để dễ kiểm tra
void fill_range(Tensor& t) {
    float* p = (float*)t.data_ptr();
    for(size_t i=0; i<t.numel(); ++i) p[i] = (float)i;
}

// Hàm điền dữ liệu bằng một số cố định
void fill_const(Tensor& t, float val) {
    float* p = (float*)t.data_ptr();
    for(size_t i=0; i<t.numel(); ++i) p[i] = val;
}

// ======================================================================
// TEST CASES
// ======================================================================

void test_dense() {
    std::cout << "\n>>> TEST: DENSE (FULLY CONNECTED) <<<" << std::endl;
    // Input: [2, 3]
    Tensor x = Tensor::zeros({2, 3}, DeviceType::CPU);
    fill_range(x); // 0 1 2; 3 4 5
    
    // Dense Layer: 3 input -> 2 output
    Dense layer(3, 2);
    
    // Gán Weight cố định để dễ nhẩm: Toàn số 1
    // W shape: [3, 2]
    fill_const(layer.W, 1.0f);
    
    // Gán Bias cố định: 0.5
    fill_const(layer.b, 0.5f);

    print_tensor_box("Input X", x);
    print_tensor_box("Weights W", layer.W);
    print_tensor_box("Bias b", layer.b);

    Tensor y = layer.forward(x);
    
    // Giải thích tính toán:
    // Hàng 0 của X: [0, 1, 2] -> Nhân với cột W (toàn 1) -> 0+1+2 = 3. Cộng bias 0.5 -> 3.5
    // Hàng 1 của X: [3, 4, 5] -> Nhân với cột W (toàn 1) -> 3+4+5 = 12. Cộng bias 0.5 -> 12.5
    print_tensor_box("Output Y", y);
}

void test_conv2d() {
    std::cout << "\n>>> TEST: CONV2D <<<" << std::endl;
    
    // Input: 1 ảnh, 1 kênh, 4x4
    Tensor x = Tensor::zeros({1, 1, 4, 4}, DeviceType::CPU);
    fill_range(x); // 0, 1, ... 15
    
    // Conv: 1 In -> 1 Out, Kernel 3x3, Stride 1, No Padding
    Conv2D layer(1, 1, 3, 1, 0);
    
    // Gán Kernel đặc biệt để dễ check:
    // 0 0 0
    // 0 1 0  <-- Chỉ giữ lại pixel ở giữa (Identity Filter)
    // 0 0 0
    // Như vậy Output phải giống hệt phần ruột của Input
    fill_const(layer.W, 0.0f);
    float* k_ptr = (float*)layer.W.data_ptr();
    // Kernel index trung tâm (1,1) là vị trí thứ 4 (0,1,2,3,4)
    k_ptr[4] = 1.0f; 
    
    fill_const(layer.b, 0.0f);

    print_tensor_box("Input Image (4x4)", x);
    print_tensor_box("Filter / Kernel (3x3)", layer.W);

    Tensor y = layer.forward(x);
    
    // Input:
    //  0  1  2  3
    //  4 [5  6] 7
    //  8 [9 10] 11
    // 12 13 14 15
    //
    // Output (Valid Conv): Sẽ lấy vùng 2x2 ở giữa:
    //  5  6
    //  9 10
    print_tensor_box("Output Feature Map", y);
}

void test_relu() {
    std::cout << "\n>>> TEST: RELU <<<" << std::endl;
    Tensor x = Tensor::zeros({1, 6}, DeviceType::CPU);
    float* p = (float*)x.data_ptr();
    p[0]=-3; p[1]=2; p[2]=0; p[3]=-0.5; p[4]=5; p[5]=-10;

    ReLU layer;
    print_tensor_box("Input", x);
    Tensor y = layer.forward(x);
    print_tensor_box("Output (No negatives)", y);
}

void test_maxpool() {
    std::cout << "\n>>> TEST: MAXPOOL2D <<<" << std::endl;
    // Input 4x4
    Tensor x = Tensor::zeros({1, 1, 4, 4}, DeviceType::CPU);
    
    // Tạo data random có chủ đích
    // Góc trái trên [0,0] -> [1,1] ta sẽ đặt max là 9
    // Góc phải dưới ta đặt max là 20
    fill_const(x, 1.0f);
    float* p = (float*)x.data_ptr();
    p[0] = 5; p[1] = 2;
    p[4] = 9; p[5] = 3; // Max vùng 2x2 đầu tiên là 9
    
    p[15] = 20.0f; // Góc cuối cùng

    MaxPool2D layer(2, 2); // Kernel 2, Stride 2 (Không chồng lấn)

    print_tensor_box("Input", x);
    Tensor y = layer.forward(x);
    print_tensor_box("Output (Max values)", y);
}

void test_upsample() {
    std::cout << "\n>>> TEST: UPSAMPLE <<<" << std::endl;
    // Input 2x2
    Tensor x = Tensor::zeros({1, 1, 2, 2}, DeviceType::CPU);
    float* p = (float*)x.data_ptr();
    p[0]=1; p[1]=2;
    p[2]=3; p[3]=4;

    Upsample layer(2); // Scale x2

    print_tensor_box("Input (2x2)", x);
    Tensor y = layer.forward(x);
    // 1 sẽ biến thành khối 2x2 số 1
    // 2 sẽ biến thành khối 2x2 số 2...
    print_tensor_box("Output (4x4)", y);
}

int main() {
    test_dense();
    test_conv2d();
    test_relu();
    test_maxpool();
    test_upsample();
    return 0;
}