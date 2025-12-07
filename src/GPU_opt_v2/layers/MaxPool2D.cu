#include "layers/MaxPool2D.h"
#include "core/CheckError.h"
#include "core/Tensor.h"
#include <cfloat>

namespace {

// ======================================================================
// 1. KERNEL FLOAT4 - 2 OUTPUTS PER THREAD
// Chiến lược: 1 Thread đọc 2 float4 (Input) -> Tính ra 2 float (Output)
// Yêu cầu: Kernel=2, Stride=2, Input Width % 4 == 0
// ======================================================================
__global__ void k_maxpool_fwd_2out_vec4(
    const float* __restrict__ in, 
    float* __restrict__ out, 
    float* __restrict__ indices,
    int C, int H_in, int W_in, 
    int H_out, int W_out) 
{
    // idx: Index của cặp đôi pixel (mỗi thread xử lý 2 pixel output liên tiếp)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Số lượng cặp pixel cần xử lý trong 1 ảnh
    // Vì mỗi thread làm 2 pixel nên số thread cần thiết = (C * H_out * W_out) / 2
    int vol_pair = (C * H_out * W_out) / 2; 
    
    // Batch index
    int n = blockIdx.z;

    if (idx >= vol_pair) return;

    // --- 1. GIẢI MÃ TỌA ĐỘ OUTPUT ---
    // Output index thực tế (của pixel đầu tiên trong cặp)
    int out_idx_0 = idx * 2;
    
    // Map linear index về (c, oh, ow)
    // Lưu ý: ow luôn là số chẵn (0, 2, 4...) do ta bước nhảy 2
    int ow = out_idx_0 % W_out;
    int tmp = out_idx_0 / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out;

    // --- 2. XÁC ĐỊNH TỌA ĐỘ INPUT ---
    // Output (oh, ow) tương ứng Input (2*oh, 2*ow)
    int h_in = oh * 2;
    int w_in = ow * 2;

    // Offset đến đầu ảnh của batch n, channel c
    int img_offset = n * (C * H_in * W_in) + c * (H_in * W_in);
    
    // Con trỏ đến điểm bắt đầu của hàng 0 và hàng 1 trong vùng 2x4
    // Ta cast sang float4* để đọc 4 phần tử liên tiếp (cột 0,1,2,3)
    const float4* row0_ptr = (const float4*)(in + img_offset + h_in * W_in + w_in);
    const float4* row1_ptr = (const float4*)(in + img_offset + (h_in + 1) * W_in + w_in);

    // --- 3. LOAD DỮ LIỆU (CHỈ 2 LẦN ĐỌC) ---
    float4 v0 = *row0_ptr; // Chứa: Row0[0], Row0[1], Row0[2], Row0[3]
    float4 v1 = *row1_ptr; // Chứa: Row1[0], Row1[1], Row1[2], Row1[3]

    // --- 4. TÍNH TOÁN POOLING (DÙNG THANH GHI) ---
    // Base index để tính chỉ số Max (cho Backward)
    int base_idx_global = img_offset + h_in * W_in + w_in; 

    // >> Output 1 (Góc trái): Input col 0, 1
    float m1 = v0.x; 
    int midx1 = 0; // Relative index: (0,0)
    
    if (v0.y > m1) { m1 = v0.y; midx1 = 1; }          // (0,1)
    if (v1.x > m1) { m1 = v1.x; midx1 = W_in; }       // (1,0)
    if (v1.y > m1) { m1 = v1.y; midx1 = W_in + 1; }   // (1,1)

    // >> Output 2 (Góc phải): Input col 2, 3
    float m2 = v0.z; 
    int midx2 = 2; // Relative index: (0,2)

    if (v0.w > m2) { m2 = v0.w; midx2 = 3; }          // (0,3)
    if (v1.z > m2) { m2 = v1.z; midx2 = W_in + 2; }   // (1,2)
    if (v1.w > m2) { m2 = v1.w; midx2 = W_in + 3; }   // (1,3)

    // --- 5. GHI KẾT QUẢ ---
    int out_global_idx = n * (C * H_out * W_out) + out_idx_0;
    
    // Ghi Value
    out[out_global_idx]     = m1;
    out[out_global_idx + 1] = m2;

    // Ghi Index
    indices[out_global_idx]     = (float)(base_idx_global + midx1);
    indices[out_global_idx + 1] = (float)(base_idx_global + midx2);
}

// ======================================================================
// KERNEL FALLBACK & BACKWARD (GIỮ NGUYÊN)
// ======================================================================
__global__ void k_maxpool_fwd_scalar(
    const float* in, float* out, float* indices,
    int C, int H, int W, int H_out, int W_out, int k, int s) 
{
    int n = blockIdx.z; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vol_one_img = C * H_out * W_out;
    
    if (idx >= vol_one_img) return;

    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    int c = tmp / H_out;

    int img_offset = n * (C * H * W);
    int h_start = oh * s;
    int w_start = ow * s;
    
    float max_val = -FLT_MAX;
    int max_idx = -1;
    const float* in_img = in + img_offset;

    for (int x = 0; x < k; ++x) {
        for (int y = 0; y < k; ++y) {
            int h_in = h_start + x;
            int w_in = w_start + y;
            if (h_in < H && w_in < W) {
                int local_idx = c*(H*W) + h_in*W + w_in;
                float val = in_img[local_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = img_offset + local_idx; 
                }
            }
        }
    }
    int out_idx = n * vol_one_img + idx;
    out[out_idx] = max_val;
    indices[out_idx] = (float)max_idx;
}

__global__ void k_maxpool_bwd_opt(const float* grad_out, const float* indices, float* grad_in, int vol_one_img) {
    int n = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vol_one_img) {
        int gid = n * vol_one_img + idx;
        int max_idx = (int)indices[gid];
        if (max_idx != -1) atomicAdd(&grad_in[max_idx], grad_out[gid]);
    }
}
} // namespace

// ======================================================================
// CLASS IMPLEMENTATION
// ======================================================================

MaxPool2D::MaxPool2D(int k, int s) : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    Tensor input_gpu = input.to(DeviceType::CUDA);
    input_shape_cache = input_gpu.sizes;

    int N = input_gpu.sizes[0]; int C = input_gpu.sizes[1]; 
    int H = input_gpu.sizes[2]; int W = input_gpu.sizes[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    Tensor out = Tensor::empty({N, C, H_out, W_out}, DeviceType::CUDA);
    indices_cache = Tensor::empty(out.sizes, DeviceType::CUDA);

    int vol_one_img = C * H_out * W_out;

    // --- KIỂM TRA ĐIỀU KIỆN DÙNG KERNEL TỐI ƯU ---
    // 1. Kernel 2x2, Stride 2
    // 2. Input Width phải chia hết cho 4 (để cast float4* an toàn tại mọi vị trí 2*ow)
    //    Hoặc chặt chẽ hơn: W_out phải chẵn (vì ta xử lý từng cặp output)
    bool is_optimized = (kernel_size == 2 && stride == 2 && (W % 4 == 0) && (W_out % 2 == 0));

    if (is_optimized) {
        // Số thread = Số cặp Output = Tổng Output / 2
        int num_pairs = vol_one_img / 2;
        int threads = 256;
        int blocks = (num_pairs + threads - 1) / threads;

        k_maxpool_fwd_2out_vec4<<<dim3(blocks, 1, N), threads>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)out.data_ptr(), 
            (float*)indices_cache.data_ptr(),
            C, H, W, H_out, W_out
        );
    } else {
        // Fallback
        int threads = 256;
        int blocks = (vol_one_img + threads - 1) / threads;
        k_maxpool_fwd_scalar<<<dim3(blocks, 1, N), threads>>>(
            (const float*)input_gpu.data_ptr(), 
            (float*)out.data_ptr(), 
            (float*)indices_cache.data_ptr(),
            C, H, W, H_out, W_out, kernel_size, stride
        );
    }
    
    CHECK(cudaGetLastError());
    return out;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_out_gpu = grad_output.to(DeviceType::CUDA);
    Tensor dX = Tensor::zeros(input_shape_cache, DeviceType::CUDA);

    int vol_img = grad_out_gpu.sizes[1] * grad_out_gpu.sizes[2] * grad_out_gpu.sizes[3];
    int N = grad_out_gpu.sizes[0];
    
    int threads = 256;
    int blocks = (vol_img + threads - 1) / threads;

    k_maxpool_bwd_opt<<<dim3(blocks, 1, N), threads>>>(
        (const float*)grad_out_gpu.data_ptr(), 
        (const float*)indices_cache.data_ptr(), 
        (float*)dX.data_ptr(), 
        vol_img
    );
    CHECK(cudaGetLastError());
    return dX;
}