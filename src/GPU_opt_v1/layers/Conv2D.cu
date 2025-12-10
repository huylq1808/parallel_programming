#include "layers/Conv2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>


__global__ void im2col_kernel(
    const float* input, float* col,
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
);

__global__ void col2im_kernel(
    const float* col, float* grad_input,
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
);

namespace {

// Cộng bias vào từng channel output
__global__ void k_add_bias(float* out, const float* bias, 
                           int Cout, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * spatial_size) return;
    
    int oc = idx / spatial_size;
    out[idx] += bias[oc];
}

// Tính gradient bias
__global__ void k_bias_grad(const float* grad_out, float* grad_b,
                            int Cout, int spatial_size) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= Cout) return;
    
    float sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        sum += grad_out[oc * spatial_size + i];
    }
    atomicAdd(&grad_b[oc], sum);
}


cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

} // namespace


Conv2D::Conv2D(int in, int out, int k, int s, int p)
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p)
{
    W = Tensor::randn({out, in, k, k}, 0.0f, 0.08f);
    b = Tensor::zeros({out});
    W.requires_grad = true;
    b.requires_grad = true;
    
    W = W.to(DeviceType::CUDA);
    b = b.to(DeviceType::CUDA);
}


Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input.to(DeviceType::CUDA);
    
    int N = input_cache.sizes[0];
    int C = input_cache.sizes[1];
    int H = input_cache.sizes[2];
    int W_in = input_cache.sizes[3];
    int H_out = (H + 2*padding - k_size) / stride + 1;
    int W_out = (W_in + 2*padding - k_size) / stride + 1;
    
    Tensor output = Tensor::zeros({N, out_c, H_out, W_out}, DeviceType::CUDA);
    
    // kích thước của ma trận cột
    int col_h = C * k_size * k_size;     // rows
    int col_w = H_out * W_out;            // cols
    
    Tensor col = Tensor::empty({col_h, col_w}, DeviceType::CUDA);
    
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    
    for (int n = 0; n < N; ++n) {
        const float* input_n = (const float*)input_cache.data_ptr() + n * C * H * W_in;
        float* output_n = (float*)output.data_ptr() + n * out_c * H_out * W_out;
        
        // 1. im2col: input [C,H,W] → col [C*K*K, H_out*W_out]
        int threads = 256;
        int total_col = col_h * col_w;
        int blocks = (total_col + threads - 1) / threads;
        im2col_kernel<<<blocks, threads>>>(
            input_n, (float*)col.data_ptr(),
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );
        
        // 2. GEMM: output = W × col
        //    W: [out_c, col_h] (row-major)
        //    col: [col_h, col_w] (row-major)
        //    output: [out_c, col_w] (row-major)
        //
        //    cuBLAS is column-major, so we compute: col^T × W^T = output^T
        //    which gives us output in row-major
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            col_w, out_c, col_h,           // m, n, k
            &alpha,
            (float*)col.data_ptr(), col_w, // A = col^T in col-major = col in row-major
            (float*)W.data_ptr(), col_h,   // B = W^T in col-major = W in row-major
            &beta,
            output_n, col_w                // C = output^T in col-major = output in row-major
        );
        
        // 3. Add bias
        int spatial = H_out * W_out;
        blocks = (out_c * spatial + threads - 1) / threads;
        k_add_bias<<<blocks, threads>>>(output_n, (float*)b.data_ptr(), out_c, spatial);
    }
    
    CHECK(cudaGetLastError());
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    Tensor grad_out = grad_output.to(DeviceType::CUDA);
    
    int N = input_cache.sizes[0];
    int C = input_cache.sizes[1];
    int H = input_cache.sizes[2];
    int W_in = input_cache.sizes[3];
    int H_out = grad_out.sizes[2];
    int W_out = grad_out.sizes[3];
    
    int col_h = C * k_size * k_size;
    int col_w = H_out * W_out;
    int spatial = H_out * W_out;
    
    // Allocate gradients
    Tensor grad_input = Tensor::zeros(input_cache.sizes, DeviceType::CUDA);
    if (!W.grad) W.ensure_grad();
    if (!b.grad) b.ensure_grad();
    W.zero_grad();
    b.zero_grad();
    
    // Buffers
    Tensor col = Tensor::empty({col_h, col_w}, DeviceType::CUDA);
    Tensor grad_col = Tensor::empty({col_h, col_w}, DeviceType::CUDA);
    
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    int threads = 256;
    
    for (int n = 0; n < N; ++n) {
        const float* input_n = (const float*)input_cache.data_ptr() + n * C * H * W_in;
        const float* grad_out_n = (const float*)grad_out.data_ptr() + n * out_c * spatial;
        float* grad_input_n = (float*)grad_input.data_ptr() + n * C * H * W_in;
        
        // 1. im2col 
        int total_col = col_h * col_w;
        int blocks = (total_col + threads - 1) / threads;
        im2col_kernel<<<blocks, threads>>>(
            input_n, (float*)col.data_ptr(),
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );
        
        // 2. dW += col × grad_out^T  (accumulate over batch)
        //    col: [col_h, col_w]
        //    grad_out^T: [col_w, out_c]
        //    dW: [col_h, out_c] but we want [out_c, col_h]
        //    So: dW^T += grad_out × col^T
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            col_h, out_c, col_w,              // m, n, k
            &alpha,
            (float*)col.data_ptr(), col_w,    // A^T
            grad_out_n, col_w,                // B
            &beta_one,                        // Accumulate!
            (float*)W.grad->data_ptr(), col_h // C
        );
        
        // 3. grad_col = W^T × grad_out
        //    W: [out_c, col_h]
        //    grad_out: [out_c, col_w]
        //    grad_col: [col_h, col_w]
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            col_w, col_h, out_c,              // m, n, k
            &alpha,
            grad_out_n, col_w,                // A
            (float*)W.data_ptr(), col_h,      // B^T
            &beta_zero,
            (float*)grad_col.data_ptr(), col_w // C
        );
        
        // 4. col2im: grad_col → grad_input
        int total_input = C * H * W_in;
        blocks = (total_input + threads - 1) / threads;
        col2im_kernel<<<blocks, threads>>>(
            (float*)grad_col.data_ptr(), grad_input_n,
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );
        
        blocks = (out_c + threads - 1) / threads;
        k_bias_grad<<<blocks, threads>>>(grad_out_n, (float*)b.grad->data_ptr(), out_c, spatial);
    }
    
    CHECK(cudaGetLastError());
    return grad_input;
}

void Conv2D::to(DeviceType device) {
    W = W.to(device);
    b = b.to(device);
}