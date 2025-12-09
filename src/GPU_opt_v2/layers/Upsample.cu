#include "layers/Upsample.h"
#include "core/CheckError.h"
#include "core/Tensor.h"

namespace {

// --- VECTORIZED FORWARD (Mỗi thread ghi 4 pixels output) ---
__global__ void k_upsample_fwd_vec4(const float* __restrict__ in, float* __restrict__ out, 
                               int C, int H_in, int W_in, int H_out, int W_out, int scale) 
{
    // idx này là index của gói float4
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Tổng số gói vector trong 1 ảnh (giả sử W_out chia hết cho 4)
    int vol_vec_one_img = (C * H_out * W_out) / 4; 

    if (idx >= vol_vec_one_img) return;

    // 1. Tính toán tọa độ Output cho phần tử đầu tiên trong gói 4
    int out_idx_0 = idx * 4; 

    // Giải mã tọa độ: ow, oh, c
    // Lưu ý: Ta giả định W_out chia hết cho 4 để ow luôn align 4
    int ow = out_idx_0 % W_out;
    int tmp = out_idx_0 / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out; 
    
    // Batch offset (Grid Z = batch size)
    int n = blockIdx.z;
    
    // Con trỏ tới ảnh input tương ứng trong batch
    const float* in_ptr = in + n * (C * H_in * W_in);
    
    // Con trỏ tới output (ép kiểu float4)
    float4* out_ptr = (float4*)out + (n * vol_vec_one_img);

    // 2. Nearest Neighbor Mapping
    int in_h = oh / scale;
    
    // Base offset của channel + row
    int base_in_offset = c * (H_in * W_in) + in_h * W_in;

    float4 res;
    // Gather: Đọc 4 giá trị input (có thể trùng nhau nếu scale lớn)
    res.x = in_ptr[base_in_offset + (ow + 0) / scale];
    res.y = in_ptr[base_in_offset + (ow + 1) / scale];
    res.z = in_ptr[base_in_offset + (ow + 2) / scale];
    res.w = in_ptr[base_in_offset + (ow + 3) / scale];

    // 3. Vectorized Store
    out_ptr[idx] = res;
}

// --- SCALAR FORWARD (Fallback) ---
__global__ void k_upsample_fwd_opt(const float* in, float* out, 
                               int C, int H, int W, int H_out, int W_out, int scale) 
{
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C * H_out * W_out;
    if (idx >= vol_one_img) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out; 

    int in_h = oh / scale;
    int in_w = ow / scale;
    
    int in_global_idx = n*(C*H*W) + c*(H*W) + in_h*W + in_w;
    int out_global_idx = n*vol_one_img + idx;
    
    out[out_global_idx] = in[in_global_idx];
}


__global__ void k_upsample_bwd_gather(
    const float* __restrict__ grad_out, 
    float* __restrict__ grad_in, 
    int C, int H_in, int W_in, 
    int H_out, int W_out, int scale) 
{
    // Mỗi thread tính 1 pixel Input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_in = C * H_in * W_in;
    
    if (idx >= vol_in) return;

    int n = blockIdx.z;

    // Giải mã tọa độ Input
    int w_in = idx % W_in;
    int tmp = idx / W_in;
    int h_in = tmp % H_in;
    int c = tmp / H_in;

    // Tọa độ gốc trên Output
    int h_out_start = h_in * scale;
    int w_out_start = w_in * scale;

    float sum = 0.0f;

    // Gom gradient từ vùng Scale x Scale
    // Với scale=2, loop này chạy 4 lần. 
    #pragma unroll
    for (int y = 0; y < scale; ++y) {
        #pragma unroll
        for (int x = 0; x < scale; ++x) {
            int h_out = h_out_start + y;
            int w_out = w_out_start + x;
            
            if (h_out < H_out && w_out < W_out) {
                int out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
                sum += grad_out[out_idx];
            }
        }
    }

    // Ghi trực tiếp (Không Atomic)
    int in_idx = n * vol_in + idx;
    grad_in[in_idx] = sum;
}

} // namespace

Upsample::Upsample(int scale) : scale_factor(scale) {}

Tensor Upsample::forward(const Tensor& input) {
    Tensor input_gpu = input.to(DeviceType::CUDA);
    input_shape_cache = input_gpu.sizes;

    int N = input_gpu.sizes[0]; int C = input_gpu.sizes[1]; 
    int H = input_gpu.sizes[2]; int W = input_gpu.sizes[3];
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;

    Tensor out = Tensor::empty({N, C, H_out, W_out}, DeviceType::CUDA);

    int vol_out = C * H_out * W_out;

    // Kiểm tra điều kiện Vectorize: Tổng số phần tử 1 ảnh chia hết cho 4
    if (vol_out % 4 == 0) {
        int vol_vec = vol_out / 4;
        int threads = 256;
        int blocks = (vol_vec + threads - 1) / threads;
        dim3 grid(blocks, 1, N);

        k_upsample_fwd_vec4<<<grid, threads>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)out.data_ptr(),
            C, H, W, H_out, W_out, scale_factor
        );
    } else {
        // Fallback: Scalar (cho trường hợp kích thước lẻ, hiếm gặp trong CNN)
        int threads = 256;
        int blocks = (vol_out + threads - 1) / threads;
        dim3 grid(blocks, 1, N);

        k_upsample_fwd_opt<<<grid, threads>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)out.data_ptr(),
            C, H, W, H_out, W_out, scale_factor
        );
    }
    
    CHECK(cudaGetLastError());
    return out;
}

Tensor Upsample::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int N = dX.sizes[0]; int C = dX.sizes[1]; int H = dX.sizes[2]; int W = dX.sizes[3];
    int H_out = grad_out_gpu.sizes[2]; int W_out = grad_out_gpu.sizes[3];

    int vol_out = C * H_out * W_out;
    int threads = 256;
    int blocks = (vol_out + threads - 1) / threads;
    dim3 grid(blocks, 1, N);

    k_upsample_bwd_gather<<<grid, threads>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (float*)dX.data_ptr(),
        C, H, W, H_out, W_out, scale_factor
    );
    CHECK(cudaGetLastError());
    return dX;
}