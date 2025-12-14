#include "layers/Conv2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel tối ưu cho im2col với batch
__global__ void im2col_kernel_batch(
    const float* input, float* col,
    int N, int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * K * K * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k_idx = (idx / (H_out * W_out)) % (C * K * K);
    int n = idx / (H_out * W_out * C * K * K);

    int c = k_idx / (K * K);
    int kh = (k_idx / K) % K;
    int kw = k_idx % K;

    int h_in = h_out * stride - pad + kh;
    int w_in = w_out * stride - pad + kw;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[n * (C * H * W) + c * (H * W) + h_in * W + w_in];
    }

    int col_idx = n * (C * K * K * H_out * W_out) + k_idx * (H_out * W_out) + h_out * W_out + w_out;
    col[col_idx] = val;
}

// Kernel tối ưu cho col2im ( grid 3D)
__global__ void col2im_kernel(
    const float* col, float* grad_input,
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int w = blockIdx.x * blockDim.x + tx;
    int h = blockIdx.y * blockDim.y + ty;
    int c = blockIdx.z;

    if (h >= H || w >= W || c >= C) return;

    float sum = 0.0f;

    int h_out_start = max(0, (h - K + 1 + pad) / stride);
    int h_out_end = min(H_out, (h + pad) / stride + 1);
    int w_out_start = max(0, (w - K + 1 + pad) / stride);
    int w_out_end = min(W_out, (w + pad) / stride + 1);

    for (int h_out = h_out_start; h_out < h_out_end; ++h_out) {
        for (int w_out = w_out_start; w_out < w_out_end; ++w_out) {
            int kh = h + pad - h_out * stride;
            int kw = w + pad - w_out * stride;
            if (kh >= 0 && kh < K && kw >= 0 && kw < K) {
                int k_idx = c * K * K + kh * K + kw;
                sum += col[k_idx * (H_out * W_out) + h_out * W_out + w_out];
            }
        }
    }

    grad_input[c * H * W + h * W + w] = sum;
}

// Kernel bias_grad với batch (sum qua N)
__global__ void k_bias_grad_batch(
    const float* grad_out, float* grad_b,
    int N, int Cout, int spatial_size
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int oc = blockIdx.x;

    if (oc >= Cout) return;

    float sum = 0.0f;
    for (int nn = 0; nn < N; ++nn) {
        for (int i = tid; i < spatial_size; i += blockDim.x) {
            int idx = nn * (Cout * spatial_size) + oc * spatial_size + i;
            sum += grad_out[idx];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(grad_b + oc, sdata[0]);
}

// add_bias với batch (loại bỏ n unused)
__global__ void k_add_bias_batch(float* out, const float* bias,
                                 int N, int Cout, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * spatial_size) return;

    int oc = (idx / spatial_size) % Cout;
    out[idx] += bias[oc];
}

// Hàm lấy cublas handle
namespace {
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

    // col cho batch đầy đủ
    long long col_h = C * k_size * k_size;
    long long col_w = H_out * W_out;
    Tensor col = Tensor::empty({N * col_h, col_w}, DeviceType::CUDA);

    // im2col batch
    int total_col = N * col_h * col_w;
    int threads = 256;
    int blocks = (total_col + threads - 1) / threads;
    im2col_kernel_batch<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), (float*)col.data_ptr(),
        N, C, H, W_in, k_size, stride, padding, H_out, W_out
    );

    // GEMM batched
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;

    long long stride_col = col_h * col_w;
    long long stride_out = out_c * col_w;
    long long stride_W = 0;

    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        col_w, out_c, col_h,
        &alpha,
        (float*)col.data_ptr(), col_w, stride_col,
        (float*)W.data_ptr(), col_h, stride_W,
        &beta,
        (float*)output.data_ptr(), col_w, stride_out,
        N
    );

    // Add bias batch
    int spatial = H_out * W_out;
    threads = 256;
    blocks = (N * out_c * spatial + threads - 1) / threads;
    k_add_bias_batch<<<blocks, threads>>>( (float*)output.data_ptr(), (float*)b.data_ptr(), N, out_c, spatial );

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

    long long col_h = C * k_size * k_size;
    long long col_w = H_out * W_out;
    int spatial = H_out * W_out;

    Tensor grad_input = Tensor::zeros(input_cache.sizes, DeviceType::CUDA);
    if (!W.grad) W.ensure_grad();
    if (!b.grad) b.ensure_grad();
    W.zero_grad();
    b.zero_grad();

    // Buffers cho batch
    Tensor col = Tensor::empty({N * col_h, col_w}, DeviceType::CUDA);
    Tensor grad_col = Tensor::empty({N * col_h, col_w}, DeviceType::CUDA);

    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    int threads = 256;

    // im2col batch (tương tự forward)
    int total_col = N * col_h * col_w;
    int blocks = (total_col + threads - 1) / threads;
    im2col_kernel_batch<<<blocks, threads>>>(
        (const float*)input_cache.data_ptr(), (float*)col.data_ptr(),
        N, C, H, W_in, k_size, stride, padding, H_out, W_out
    );

    // dW: col^T * grad_out (batched, accumulate)
    long long stride_col = col_h * col_w;
    long long stride_grad_out = out_c * col_w;
    long long stride_dW = 0;

    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        col_h, out_c, col_w,
        &alpha,
        (float*)col.data_ptr(), col_w, stride_col,
        (float*)grad_out.data_ptr(), col_w, stride_grad_out,
        &beta_one,
        (float*)W.grad->data_ptr(), col_h, stride_dW,
        N
    );

    // grad_col = grad_out * W^T (đổi OP_N, OP_T để đúng)
    long long stride_W = 0;
    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        col_w, col_h, out_c,
        &alpha,
        (float*)grad_out.data_ptr(), col_w, stride_grad_out,
        (float*)W.data_ptr(), col_h, stride_W,
        &beta_zero,
        (float*)grad_col.data_ptr(), col_w, stride_col,
        N
    );

    // col2im: grid 3D cho parallelism
    dim3 threads_dim(16, 16);
    dim3 blocks_dim((W_in + 15)/16, (H + 15)/16, C);
    col2im_kernel<<<blocks_dim, threads_dim>>>(
        (float*)grad_col.data_ptr(), (float*)grad_input.data_ptr(),
        C, H, W_in, k_size, stride, padding, H_out, W_out
    );

    // Bias grad batch
    threads = 1024;
    blocks = out_c;
    size_t shm_size = 1024 * sizeof(float);
    k_bias_grad_batch<<<blocks, threads, shm_size>>>(
        (const float*)grad_out.data_ptr(), (float*)b.grad->data_ptr(), N, out_c, spatial
    );

    CHECK(cudaGetLastError());
    return grad_input;
}

void Conv2D::to(DeviceType device) {
    W = W.to(device);
    b = b.to(device);
}