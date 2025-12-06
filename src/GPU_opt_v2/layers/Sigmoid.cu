#include "layers/Sigmoid.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

__global__ void k_sigmoid_fwd_opt(const float* in, float* out, int vol_one_img) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vol_one_img) {
        int gid = n * vol_one_img + idx;
        float x = in[gid];
        if (x > 88.0f) out[gid] = 1.0f;
        else if (x < -88.0f) out[gid] = 0.0f;
        else out[gid] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void k_sigmoid_bwd_opt(const float* y, const float* go, float* gi, int vol_one_img) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vol_one_img) {
        int gid = n * vol_one_img + idx;
        // dL/dx = dL/dy * y * (1 - y)
        gi[gid] = go[gid] * y[gid] * (1.0f - y[gid]);
    }
}

} // namespace

Tensor Sigmoid::forward(const Tensor& input) {
    
    Tensor input_gpu = input.to(DeviceType::CUDA);

    if (output_cache.sizes != input_gpu.sizes || output_cache.device != DeviceType::CUDA) {
        output_cache = Tensor::empty(input_gpu.sizes, DeviceType::CUDA);
    }

    int n = input_gpu.numel();
    if (n == 0) return output_cache;

    // 3. Launch Kernel với con trỏ GPU chắc chắn
    k_sigmoid_fwd_opt<<<(n + 255)/256, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)output_cache.data_ptr(), 
        n
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    
    return output_cache;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {

    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    
    Tensor grad_input = Tensor::empty(output_cache.sizes, DeviceType::CUDA);

    int n = output_cache.numel();
    
    k_sigmoid_bwd_opt<<<(n + 255)/256, 256>>>(
        (const float*)output_cache.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)grad_input.data_ptr(), 
        n
    );
    CHECK(cudaGetLastError());
    //CHECK(cudaDeviceSynchronize());
    return grad_input;
}