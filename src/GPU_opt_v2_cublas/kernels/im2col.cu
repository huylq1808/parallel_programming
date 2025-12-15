// đưa input sang ma trận cột
__global__ void im2col_kernel(
    const float* input,      // [N, C, H, W]
    float* col,              // [N, C*K*K, H_out*W_out]
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    // Mỗi thread xử lý 1 phần tử
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * K * K * H_out * W_out;
    if (idx >= total) return;
    
    // Decode index
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k_idx = idx / (H_out * W_out);  // which element in the C*K*K patch
    
    int c = k_idx / (K * K);
    int kh = (k_idx / K) % K;
    int kw = k_idx % K;
    
    int h_in = h_out * stride - pad + kh;
    int w_in = w_out * stride - pad + kw;
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[c * H * W + h_in * W + w_in];
    }
    
    col[k_idx * (H_out * W_out) + h_out * W_out + w_out] = val;
}

// Backward: col2im - fold gradients back to input shape
__global__ void col2im_kernel(
    const float* col,        // [N, C*K*K, H_out*W_out]
    float* grad_input,       // [N, C, H, W]
    int C, int H, int W,
    int K, int stride, int pad,
    int H_out, int W_out
) {
    // Each thread handles one input pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;
    
    int w = idx % W;
    int h = (idx / W) % H;
    int c = idx / (H * W);
    
    float sum = 0.0f;
    
    // Find all output positions that used this input pixel
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int h_out = (h + pad - kh);
            int w_out = (w + pad - kw);
            
            if (h_out % stride == 0 && w_out % stride == 0) {
                h_out /= stride;
                w_out /= stride;
                
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int k_idx = c * K * K + kh * K + kw;
                    sum += col[k_idx * (H_out * W_out) + h_out * W_out + w_out];
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}