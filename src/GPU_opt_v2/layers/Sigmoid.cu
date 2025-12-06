#include "layers/Sigmoid.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

// --- HELPER FUNCTIONS ---
__device__ __forceinline__ float sigmoid_op(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    return 1.0f / (1.0f + __expf(-x)); // __expf nhanh hơn expf thường
}

__device__ __forceinline__ float sigmoid_grad_op(float y, float go) {
    return go * y * (1.0f - y);
}

// --- VECTORIZED KERNELS (FLOAT4) ---
__global__ void k_sigmoid_fwd_vec4(const float* __restrict__ in, float* __restrict__ out, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // Ép kiểu sang float4
    const float4* in_vec = (const float4*)in;
    float4* out_vec = (float4*)out;

    float4 v = in_vec[idx]; // Load 128-bit
    float4 r;

    r.x = sigmoid_op(v.x);
    r.y = sigmoid_op(v.y);
    r.z = sigmoid_op(v.z);
    r.w = sigmoid_op(v.w);

    out_vec[idx] = r; // Store 128-bit
}

__global__ void k_sigmoid_bwd_vec4(const float* __restrict__ y, const float* __restrict__ go, float* __restrict__ gi, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    const float4* y_vec = (const float4*)y;
    const float4* go_vec = (const float4*)go;
    float4* gi_vec = (float4*)gi;

    float4 v_y = y_vec[idx];
    float4 v_go = go_vec[idx];
    float4 v_gi;

    v_gi.x = sigmoid_grad_op(v_y.x, v_go.x);
    v_gi.y = sigmoid_grad_op(v_y.y, v_go.y);
    v_gi.z = sigmoid_grad_op(v_y.z, v_go.z);
    v_gi.w = sigmoid_grad_op(v_y.w, v_go.w);

    gi_vec[idx] = v_gi;
}

// --- SCALAR KERNELS (Xử lý phần dư) ---
__global__ void k_sigmoid_fwd_scalar(const float* in, float* out, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        out[idx] = sigmoid_op(in[idx]);
    }
}

__global__ void k_sigmoid_bwd_scalar(const float* y, const float* go, float* gi, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        gi[idx] = sigmoid_grad_op(y[idx], go[idx]);
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

    // 1. Chạy phần Vectorized (chia hết cho 4)
    int vec_size = n / 4;
    if (vec_size > 0) {
        int threads = 256;
        int blocks = (vec_size + threads - 1) / threads;
        k_sigmoid_fwd_vec4<<<blocks, threads>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)output_cache.data_ptr(), 
            vec_size
        );
    }

    // 2. Chạy phần dư (Tail)
    int remainder = n % 4;
    if (remainder > 0) {
        int offset = vec_size * 4;
        k_sigmoid_fwd_scalar<<<1, remainder>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)output_cache.data_ptr(), 
            n, offset
        );
    }
    
    CHECK(cudaGetLastError());
    return output_cache;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor grad_input = Tensor::empty(output_cache.sizes, DeviceType::CUDA);

    int n = output_cache.numel();
    
    // 1. Vectorized Backward
    int vec_size = n / 4;
    if (vec_size > 0) {
        int threads = 256;
        int blocks = (vec_size + threads - 1) / threads;
        k_sigmoid_bwd_vec4<<<blocks, threads>>>(
            (const float*)output_cache.data_ptr(), 
            (const float*)grad_out_gpu.data_ptr(), 
            (float*)grad_input.data_ptr(), 
            vec_size
        );
    }

    // 2. Scalar Backward
    int remainder = n % 4;
    if (remainder > 0) {
        int offset = vec_size * 4;
        k_sigmoid_bwd_scalar<<<1, remainder>>>(
            (const float*)output_cache.data_ptr(), 
            (const float*)grad_out_gpu.data_ptr(), 
            (float*)grad_input.data_ptr(), 
            n, offset
        );
    }

    CHECK(cudaGetLastError());
    return grad_input;
}