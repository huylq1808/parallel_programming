#include "NaiveOps.cuh"
#include <iostream>
#include <cuda_runtime.h>

__global__ void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void cuda_launch_add_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    int n = A.numel();
    k_add<<<(n + 255) / 256, 256>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), n);
    CHECK(cudaGetLastError());
}

// MATMUL NAIVE IMPLEMENTATION
// Formula: C[row, col] = Sum(A[row, k] * B[k, col])
// ============================================================
__global__ void k_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // Tính chỉ số hàng (row) và cột (col) của ma trận kết quả C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Duyệt qua chiều chung K để tính tích vô hướng
        for (int k = 0; k < K; ++k) {
            // A: Row-major [M x K] -> index = row * K + k
            // B: Row-major [K x N] -> index = k * N + col
            sum += A[row * K + k] * B[k * N + col];
        }
        // Ghi kết quả vào C [M x N]
        C[row * N + col] = sum;
    }
}

void cuda_launch_matmul_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    // 1. Kiểm tra kích thước (Giả sử Tensor là 2D)
    // A: [M, K], B: [K, N], C: [M, N]
    if (A.sizes.size() != 2 || B.sizes.size() != 2) {
        std::cerr << "Error: MatMul Naive only supports 2D Tensors.\n";
        return;
    }

    int M = A.sizes[0];
    int K = A.sizes[1];
    int N = B.sizes[1];

    if (B.sizes[0] != K) {
        std::cerr << "Error: MatMul shape mismatch (A cols != B rows).\n";
        return;
    }

    // 2. Cấu hình Grid/Block 2 chiều
    // Block 16x16 threads (256 threads per block)
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x, // Grid X bao phủ số cột N
        (M + block.y - 1) / block.y  // Grid Y bao phủ số hàng M
    );

    // 3. Gọi Kernel
    k_matmul_naive<<<grid, block>>>(
        (float*)A.data_ptr(), 
        (float*)B.data_ptr(), 
        (float*)C.data_ptr(), 
        M, N, K
    );
    CHECK(cudaGetLastError());
}