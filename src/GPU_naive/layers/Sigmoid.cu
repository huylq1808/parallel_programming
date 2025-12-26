#include "layers/Sigmoid.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

__global__ void k_sigmoid_fwd(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        if (x > 88.0f) out[i] = 1.0f;
        else if (x < -88.0f) out[i] = 0.0f;
        else out[i] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void k_sigmoid_bwd(const float* y, const float* go, float* gi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // dL/dx = dL/dy * y * (1 - y)
        gi[i] = go[i] * y[i] * (1.0f - y[i]);
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
    k_sigmoid_fwd<<<(n + 255)/256, 256>>>(
        (const float*)input_gpu.data_ptr(), 
        (float*)output_cache.data_ptr(), 
        n
    );
    CHECK(cudaGetLastError());
    
    return output_cache;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {

    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    
    Tensor grad_input = Tensor::empty(output_cache.sizes, DeviceType::CUDA);

    int n = output_cache.numel();
    
    k_sigmoid_bwd<<<(n + 255)/256, 256>>>(
        (const float*)output_cache.data_ptr(), 
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)grad_input.data_ptr(), 
        n
    );
    CHECK(cudaGetLastError());
    
    return grad_input;
}