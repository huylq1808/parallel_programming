#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "../include/core/Tensor.h"

// --- TEST UTILS ---

#define ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        std::cerr << "   [FAIL] " << msg << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } else { \
        std::cout << "   [PASS] " << msg << std::endl; \
    }

bool is_close(float a, float b, float eps = 1e-4) {
    return std::fabs(a - b) < eps;
}

// Hàm in Tensor xịn xò hơn để theo dõi
void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << "\n--- Viewing: " << name << " ---" << std::endl;
    std::cout << "Meta: Shape=[";
    for(size_t i=0; i<t.sizes.size(); ++i) std::cout << t.sizes[i] << (i<t.sizes.size()-1?",":"");
    std::cout << "], Strides=[";
    for(size_t i=0; i<t.strides.size(); ++i) std::cout << t.strides[i] << (i<t.strides.size()-1?",":"");
    std::cout << "]" << std::endl;

    const float* ptr = (const float*)t.data_ptr();

    if (t.sizes.size() == 1) {
        // In Vector 1D
        std::cout << "Data: [ ";
        for (int i = 0; i < t.sizes[0]; ++i) {
             // Offset logic cho 1D
            int offset = i * t.strides[0];
            std::cout << std::fixed << std::setprecision(1) << ptr[offset] << " ";
        }
        std::cout << "]" << std::endl;
    }
    else if (t.sizes.size() == 2) {
        // In Ma trận 2D
        std::cout << "Data:" << std::endl;
        for (int i = 0; i < t.sizes[0]; ++i) {
            std::cout << "  ";
            for (int j = 0; j < t.sizes[1]; ++j) {
                // Offset logic cho 2D (quan trọng để check transpose)
                int offset = i * t.strides[0] + j * t.strides[1];
                std::cout << std::fixed << std::setprecision(1) << std::setw(5) << ptr[offset] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        // Với 3D trở lên, in flatten data để check nhanh
        std::cout << "Data (Flattened View): [ ";
        // Lưu ý: In kiểu này chỉ đúng nếu contiguous, nhưng tạm dùng để debug giá trị
        size_t n = t.numel();
        for (size_t i = 0; i < std::min(n, (size_t)20); ++i) { 
            std::cout << ptr[i] << " ";
        }
        if (n > 20) std::cout << "... ";
        std::cout << "]" << std::endl;
    }
    std::cout << "-----------------------------------" << std::endl;
}

// --- TEST CASES ---

void test_initialization() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "TEST 1: Initialization (Zeros)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    Tensor t = Tensor::zeros({2, 3}, DeviceType::CPU);
    print_tensor("Zeros Tensor", t);
    
    float* ptr = (float*)t.data_ptr();
    ASSERT_TRUE(t.numel() == 6, "Numel is 6");
    ASSERT_TRUE(ptr[0] == 0.0f && ptr[5] == 0.0f, "Values are 0.0");
}

void test_transpose_view_logic() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "TEST 2: Transpose Logic (Memory View)" << std::endl;
    std::cout << "=============================================" << std::endl;

    // 1. Tạo Tensor
    Tensor t = Tensor::zeros({2, 3}, DeviceType::CPU);
    float* data = (float*)t.data_ptr();
    // Gán giá trị 1..6
    data[0]=1; data[1]=2; data[2]=3;
    data[3]=4; data[4]=5; data[5]=6;

    print_tensor("Original Matrix (2x3)", t);

    // 2. Transpose
    Tensor t_T = t.transpose(0, 1);
    print_tensor("Transposed Matrix (3x2)", t_T);

    // 3. Checks
    ASSERT_TRUE(t_T.sizes[0] == 3 && t_T.sizes[1] == 2, "Shape swapped");
    ASSERT_TRUE(t.data_ptr() == t_T.data_ptr(), "Memory address is identical (Zero-copy)");
    
    // Check giá trị tại (1,0) của ma trận mới (phải là số 2.0)
    // Offset logic bên trong print_tensor đã in ra đúng vị trí thị giác
    // Code check logic:
    const float* ptr_T = (const float*)t_T.data_ptr();
    float val_at_1_0 = ptr_T[1 * t_T.strides[0] + 0 * t_T.strides[1]]; // Hàng 1, Cột 0
    ASSERT_TRUE(val_at_1_0 == 2.0f, "Value at [1,0] corresponds to original [0,1]");
}

void test_matmul_correctness() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "TEST 3: Matmul Mathematics" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Matrix A (2x2)
    Tensor A = Tensor::zeros({2, 2}, DeviceType::CPU);
    float* dA = (float*)A.data_ptr();
    dA[0]=1; dA[1]=2; 
    dA[2]=3; dA[3]=4;
    print_tensor("Matrix A", A);

    // Matrix B (Identity 2x2)
    Tensor B = Tensor::zeros({2, 2}, DeviceType::CPU);
    float* dB = (float*)B.data_ptr();
    dB[0]=1; dB[1]=0; 
    dB[2]=0; dB[3]=1;
    print_tensor("Matrix B (Identity)", B);

    // C = A * B
    Tensor C = A.matmul(B);
    print_tensor("Result C = A * B", C);

    const float* dC = (const float*)C.data_ptr();
    ASSERT_TRUE(dC[0]==1 && dC[1]==2 && dC[2]==3 && dC[3]==4, "A * I == A");

    // Test Dot Product (Rectangular)
    std::cout << "\n--- Sub-test: Dot Product ---" << std::endl;
    Tensor V1 = Tensor::zeros({1, 3}, DeviceType::CPU);
    float* dV1 = (float*)V1.data_ptr();
    dV1[0]=1; dV1[1]=2; dV1[2]=3;
    print_tensor("Vector 1 (1x3)", V1);

    Tensor V2 = Tensor::zeros({3, 1}, DeviceType::CPU);
    float* dV2 = (float*)V2.data_ptr();
    dV2[0]=1; dV2[1]=1; dV2[2]=1;
    print_tensor("Vector 2 (3x1)", V2);

    Tensor Res = V1.matmul(V2);
    print_tensor("Result (Scalar)", Res);
    
    const float* dR = (const float*)Res.data_ptr();
    // 1*1 + 2*1 + 3*1 = 6
    ASSERT_TRUE(is_close(dR[0], 6.0f), "Dot product is 6.0");
}

void test_complex_chain() {
    std::cout << "\n=============================================" << std::endl;
    std::cout << "TEST 4: Complex Chaining (Flatten -> View)" << std::endl;
    std::cout << "=============================================" << std::endl;

    // 1. Tạo 3D Tensor [2, 2, 2]
    Tensor t = Tensor::zeros({2, 2, 2}, DeviceType::CPU);
    float* dt = (float*)t.data_ptr();
    for(int i=0; i<8; ++i) dt[i] = (float)i; // 0, 1, 2...7
    print_tensor("Original 3D [2,2,2]", t);

    // 2. Flatten -> [2, 4]
    Tensor flat = t.flatten();
    print_tensor("Flattened [2, 4]", flat);
    ASSERT_TRUE(flat.sizes[0]==2 && flat.sizes[1]==4, "Flatten shape correct");

    // 3. View -> [8]
    Tensor v1 = flat.view({8});
    print_tensor("View 1D [8]", v1);
    ASSERT_TRUE(v1.sizes[0]==8, "View 1D shape correct");

    // 4. View back -> [4, 2]
    Tensor v2 = v1.view({4, 2});
    print_tensor("Reshape to [4, 2]", v2);
    ASSERT_TRUE(v2.sizes[0]==4 && v2.sizes[1]==2, "Reshape correct");
    
    // Check data consistency
    // v2[3, 1] là phần tử cuối cùng (index 7), giá trị phải là 7.0
    const float* dv2 = (const float*)v2.data_ptr();
    // Offset = 3*stride[0] + 1*stride[1] = 3*1 + 1*1 = 4 ??? KHÔNG ĐÚNG
    // Stride của [4, 2] là [2, 1]
    // Offset = 3*2 + 1*1 = 7. Đúng!
    ASSERT_TRUE(dv2[7] == 7.0f, "Data integrity maintained");
}

int main() {
    try {
        test_initialization();
        test_transpose_view_logic();
        test_matmul_correctness();
        test_complex_chain();
        std::cout << "\n>>> ALL TESTS PASSED! CODE IS WORKING PERFECTLY. <<<" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}