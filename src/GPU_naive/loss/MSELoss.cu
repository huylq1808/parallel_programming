#include "loss/MSELoss.h"
#include "core/Tensor.h"
#include "core/CheckError.h"
#include <cuda_runtime.h>

namespace {

__global__ void k_mse_loss(const float* pred, const float* target, float* shared_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = pred[i] - target[i];
        atomicAdd(shared_sum, diff * diff);
    }
}

__global__ void k_mse_bwd(const float* pred, const float* target, float* grad, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] = scale * (pred[i] - target[i]);
    }
}

} // namespace

float MSELoss::forward(const Tensor& input, const Tensor& target) {
    // 1. [QUAN TRỌNG] Đảm bảo cả input và target đều ở trên GPU
    // Input từ model ra thì đã là GPU, nhưng Target từ DataLoader vào thường là CPU.
    input_cache = input; // (Giả sử input đã ở GPU, nếu chưa chắc thì .to(CUDA))
    target_cache = target.to(DeviceType::CUDA); // <--- FIX LỖI 700 TẠI ĐÂY

    int n = input_cache.numel();
    
    // 2. Cấp phát biến tổng lỗi trên GPU
    float* d_sum;
    CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // 3. Chạy kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    k_mse_loss<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)target_cache.data_ptr(), // Bây giờ đây là pointer GPU an toàn
        d_sum, 
        n
    );
    CHECK(cudaGetLastError());

    // 4. Copy kết quả về CPU
    float h_sum;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sum));

    return h_sum / n;
}

Tensor MSELoss::backward() {
    // 5. [QUAN TRỌNG] Tạo Tensor gradient trên GPU
    // Nếu không chỉ định DeviceType::CUDA, nó sẽ tạo trên CPU -> Crash khi đưa vào kernel
    Tensor dInput = Tensor::zeros(input_cache.sizes, DeviceType::CUDA); 

    int n = input_cache.numel();
    float scale = 2.0f / n;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    k_mse_bwd<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)target_cache.data_ptr(), 
        (float*)dInput.data_ptr(), // Pointer GPU
        n, 
        scale
    );
    CHECK(cudaGetLastError());

    return dInput;
}