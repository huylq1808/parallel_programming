#include "layers/Conv2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ================================================================
// im2col / col2im kernels
// ================================================================

__global__ void im2col_kernel(
    const float* input, float* col,
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * K * K * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k_w = (idx / (W_out * H_out)) % K;
    int k_h = (idx / (W_out * H_out * K)) % K;
    int c = idx / (W_out * H_out * K * K);

    int h_in = h_out * stride - pad + k_h;
    int w_in = w_out * stride - pad + k_w;

    int col_row = c * K * K + k_h * K + k_w;
    int col_col = h_out * W_out + w_out;

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        col[col_row * (H_out * W_out) + col_col] =
            input[(c * H + h_in) * W + w_in];
    else
        col[col_row * (H_out * W_out) + col_col] = 0.0f;
}

__global__ void col2im_kernel(
    const float* col, float* grad_input,
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = idx / (H * W);

    float val = 0.0f;

    for (int kh = 0; kh < K; ++kh)
        for (int kw = 0; kw < K; ++kw) {
            int h_out = (h + pad - kh);
            int w_out = (w + pad - kw);
            if (h_out % stride == 0 && w_out % stride == 0) {
                h_out /= stride;
                w_out /= stride;
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int col_row = c * K * K + kh * K + kw;
                    int col_col = h_out * W_out + w_out;
                    val += col[col_row * (H_out * W_out) + col_col];
                }
            }
        }

    grad_input[idx] += val;
}

// ================================================================
// Naive GEMM kernel: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
// ================================================================

__global__ void gemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    bool transA, bool transB,
    bool accumulate
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = accumulate ? C[row * ldc + col] : 0.0f;
    
    for (int k = 0; k < K; ++k) {
        float a_val = transA ? A[k * lda + row] : A[row * lda + k];
        float b_val = transB ? B[col * ldb + k] : B[k * ldb + col];
        sum += a_val * b_val;
    }
    
    C[row * ldc + col] = sum;
}

// ================================================================
// Bias kernels
// ================================================================

__global__ void k_add_bias(float* out, const float* bias,
                           int Cout, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * spatial) return;
    int oc = idx / spatial;
    out[idx] += bias[oc];
}

__global__ void k_bias_grad(const float* grad_out, float* grad_b,
                            int Cout, int spatial) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= Cout) return;
    float sum = 0.0f;
    for (int i = 0; i < spatial; ++i)
        sum += grad_out[oc * spatial + i];
    atomicAdd(&grad_b[oc], sum);
}

// ================================================================
// Conv2D implementation
// ================================================================

Conv2D::Conv2D(int in, int out, int k, int s, int p)
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p) {

    W = Tensor::randn({out, in, k, k}, 0.0f, 0.08f).to(DeviceType::CUDA);
    b = Tensor::zeros({out}).to(DeviceType::CUDA);
    W.requires_grad = true;
    b.requires_grad = true;
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input.to(DeviceType::CUDA);

    int N = input_cache.sizes[0];
    int C = input_cache.sizes[1];
    int H = input_cache.sizes[2];
    int W_in = input_cache.sizes[3];

    int H_out = (H + 2 * padding - k_size) / stride + 1;
    int W_out = (W_in + 2 * padding - k_size) / stride + 1;

    Tensor output = Tensor::zeros({N, out_c, H_out, W_out}, DeviceType::CUDA);

    int col_h = C * k_size * k_size;
    int col_w = H_out * W_out;

    Tensor col = Tensor::empty({col_h, col_w}, DeviceType::CUDA);

    dim3 block(16, 16);
    dim3 grid((col_w + 15) / 16, (out_c + 15) / 16);

    for (int n = 0; n < N; ++n) {
        im2col_kernel<<<(col_h * col_w + 255) / 256, 256>>>(
            (float*)input_cache.data_ptr() + n * C * H * W_in,
            (float*)col.data_ptr(),
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );

        gemm_naive<<<grid, block>>>(
            (float*)W.data_ptr(),
            (float*)col.data_ptr(),
            (float*)output.data_ptr() + n * out_c * col_w,
            out_c, col_w, col_h,
            false
        );

        k_add_bias<<<(out_c * col_w + 255) / 256, 256>>>(
            (float*)output.data_ptr() + n * out_c * col_w,
            (float*)b.data_ptr(),
            out_c, col_w
        );
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

    Tensor grad_input = Tensor::zeros(input_cache.sizes, DeviceType::CUDA);
    W.ensure_grad(); W.zero_grad();
    b.ensure_grad(); b.zero_grad();

    Tensor col = Tensor::empty({col_h, col_w}, DeviceType::CUDA);
    Tensor grad_col = Tensor::zeros({col_h, col_w}, DeviceType::CUDA);

    dim3 block(16, 16);

    for (int n = 0; n < N; ++n) {
        im2col_kernel<<<(col_h * col_w + 255) / 256, 256>>>(
            (float*)input_cache.data_ptr() + n * C * H * W_in,
            (float*)col.data_ptr(),
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );

        dim3 grid_w((out_c + 15) / 16, (col_h + 15) / 16);
        gemm_naive<<<grid_w, block>>>(
            (float*)grad_out.data_ptr() + n * out_c * col_w,
            (float*)col.data_ptr(),
            (float*)W.grad->data_ptr(),
            out_c, col_h, col_w,
            col_w, col_w, col_h,
            false, true,
            true
        );

        dim3 grid_col((col_w + 15) / 16, (col_h + 15) / 16);
        gemm_naive<<<grid_col, block>>>(
            (float*)W.data_ptr(),
            (float*)grad_out.data_ptr() + n * out_c * col_w,
            (float*)grad_col.data_ptr(),
            col_h, col_w, out_c,
            col_h, col_w, col_w,
            true, false,
            false
        );

        col2im_kernel<<<(C * H * W_in + 255) / 256, 256>>>(
            (float*)grad_col.data_ptr(),
            (float*)grad_input.data_ptr() + n * C * H * W_in,
            C, H, W_in, k_size, stride, padding, H_out, W_out
        );

        k_bias_grad<<<(out_c + 255) / 256, 256>>>(
            (float*)grad_out.data_ptr() + n * out_c * col_w,
            (float*)b.grad->data_ptr(),
            out_c, col_w
        );
    }

    CHECK(cudaGetLastError());
    return grad_input;
}

void Conv2D::to(DeviceType device) {
    W = W.to(device);
    b = b.to(device);
}
