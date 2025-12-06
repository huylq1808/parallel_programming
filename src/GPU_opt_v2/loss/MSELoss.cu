#include "loss/MSELoss.h"
#include "core/Tensor.h"
#include "core/CheckError.h"
#include <cuda_runtime.h>

namespace {

__global__ void k_mse_loss_opt(const float* pred, const float* target, float* global_loss, int vol_one_img) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (idx < vol_one_img) {
        int gid = n * vol_one_img + idx;
        float diff = pred[gid] - target[gid];
        local_sum = diff * diff;
    }

    // --- BLOCK REDUCTION ---
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    // Dùng atomicAdd vào Shared Memory (Nhanh hơn Global)
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    // Ghi kết quả Block ra Global (Chỉ 1 thread làm)
    if (threadIdx.x == 0) {
        atomicAdd(global_loss, s_sum);
    }
}

__global__ void k_mse_bwd_opt(const float* pred, const float* target, float* grad, int vol_one_img, float scale) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vol_one_img) {
        int gid = n * vol_one_img + idx;
        grad[gid] = scale * (pred[gid] - target[gid]);
    }
}

} // namespace

float MSELoss::forward(const Tensor& input, const Tensor& target) {
    // 1. Lưu Input và Target (Đảm bảo Target ở trên GPU)
    input_cache = input.to(DeviceType::CUDA);
    target_cache = target.to(DeviceType::CUDA); 

    int N = input.sizes[0];
    int vol_one_img = input.numel() / N; // Số pixel trong 1 ảnh

    // 2. Cấp phát biến tổng lỗi (Accumulator) trên GPU
    float* d_sum;
    CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // 3. Cấu hình Grid 3D
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N); // Z = Batch Size

    // 4. Chạy Kernel
    k_mse_loss_opt<<<grid, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)target_cache.data_ptr(), 
        d_sum, 
        vol_one_img
    );
    CHECK(cudaGetLastError());
    ////CHECK(cudaDeviceSynchronize());

    // 5. Copy kết quả tổng về CPU
    float h_sum;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sum));

    // Trả về trung bình lỗi (Mean Squared Error)
    return h_sum / input.numel();
}

Tensor MSELoss::backward() {
    // 1. Tạo Tensor gradient đầu ra trên GPU
    Tensor dInput = Tensor::empty(input_cache.sizes, DeviceType::CUDA);

    int N = input_cache.sizes[0];
    int vol_one_img = input_cache.numel() / N;
    float scale = 2.0f / input_cache.numel();

    // 2. Cấu hình Grid 3D
    int threads = 256;
    int blocks = (vol_one_img + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    // 3. Chạy Kernel Backward
    k_mse_bwd_opt<<<grid, threads>>>(
        (const float*)input_cache.data_ptr(), 
        (const float*)target_cache.data_ptr(), 
        (float*)dInput.data_ptr(), 
        vol_one_img, 
        scale
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());

    return dInput;
}