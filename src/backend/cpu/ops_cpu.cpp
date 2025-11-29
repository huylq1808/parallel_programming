// this file is contain implementation for operation on CPU
#include <cstring>
#include <iostream>
#include <cmath>
#include "../../../include/core/Ops.h"

void cpu_matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.sizes[0]; int K = A.sizes[1]; int N = B.sizes[1];
    const float* a_ptr = (const float*)A.data_ptr();
    const float* b_ptr = (const float*)B.data_ptr();
    float* c_ptr = (float*)C.data_ptr();

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a_ptr[m * A.strides[0] + k * A.strides[1]] * b_ptr[k * B.strides[0] + n * B.strides[1]];
            }
            c_ptr[m * C.strides[0] + n * C.strides[1]] = sum;
        }
    }
}


// ================= Conv2D =================
void cpu_conv2d(const Tensor& in, const Tensor& k, const Tensor& b, Tensor& out, int stride, int padding) {
    // 1. Lấy kích thước
    int N = in.sizes[0];
    int C_in = in.sizes[1];
    int H_in = in.sizes[2];
    int W_in = in.sizes[3]; // Đổi tên thành W_in cho rõ ràng

    int C_out = k.sizes[0];
    int K_size = k.sizes[2];
    
    int H_out = out.sizes[2];
    int W_out = out.sizes[3];

    // 2. Lấy con trỏ dữ liệu (Đổi tên biến để tránh lỗi int vs pointer)
    const float* in_ptr = (const float*)in.data_ptr();
    const float* w_ptr = (const float*)k.data_ptr();
    const float* b_ptr = (const float*)b.data_ptr();
    float* out_ptr = (float*)out.data_ptr();

    // 3. Loop tính toán
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    
                    // Khởi tạo giá trị bằng Bias
                    float sum = b_ptr[oc];

                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_size; ++kh) {
                            for (int kw = 0; kw < K_size; ++kw) {
                                
                                int hi = oh * stride - padding + kh;
                                int wi = ow * stride - padding + kw;

                                // Kiểm tra biên (Padding check)
                                if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    // Index Input: [n, ic, hi, wi]
                                    int in_idx = n * (C_in * H_in * W_in) + 
                                                 ic * (H_in * W_in) + 
                                                 hi * W_in + wi;
                                    
                                    // Index Weight: [oc, ic, kh, kw]
                                    int w_idx = oc * (C_in * K_size * K_size) + 
                                                ic * (K_size * K_size) + 
                                                kh * K_size + kw;
                                    
                                    // Tính tích chập
                                    sum += in_ptr[in_idx] * w_ptr[w_idx];
                                }
                            }
                        }
                    }
                    
                    // Gán kết quả Output
                    int out_idx = n * (C_out * H_out * W_out) + 
                                  oc * (H_out * W_out) + 
                                  oh * W_out + ow;
                    out_ptr[out_idx] = sum;
                }
            }
        }
    }
}

void cpu_conv2d_backward(const Tensor& in, const Tensor& k, const Tensor& grad_out, 
                         Tensor& grad_in, Tensor& grad_k, Tensor& grad_b, int stride, int padding) {
    
    // 1. Lấy kích thước
    int N = in.sizes[0];
    int C_in = in.sizes[1];
    int H_in = in.sizes[2];
    int W_in = in.sizes[3];

    int C_out = k.sizes[0];
    int K_size = k.sizes[2]; // Giả sử kernel vuông KxK

    int H_out = grad_out.sizes[2];
    int W_out = grad_out.sizes[3];

    // 2. Reset toàn bộ gradient về 0 (Rất quan trọng vì ta dùng phép cộng dồn +=)
    std::memset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float));
    std::memset(grad_k.data_ptr(), 0, grad_k.numel() * sizeof(float));
    std::memset(grad_b.data_ptr(), 0, grad_b.numel() * sizeof(float));

    // 3. Lấy con trỏ dữ liệu
    const float* in_ptr = (const float*)in.data_ptr();
    const float* w_ptr = (const float*)k.data_ptr();
    const float* go_ptr = (const float*)grad_out.data_ptr();

    float* gi_ptr = (float*)grad_in.data_ptr();
    float* gk_ptr = (float*)grad_k.data_ptr();
    float* gb_ptr = (float*)grad_b.data_ptr();

    // 4. Loop tính toán (7 vòng for lồng nhau)
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    
                    // Lấy giá trị gradient tại pixel output hiện tại (Gradient từ layer sau truyền về)
                    int go_idx = n * (C_out * H_out * W_out) + 
                                 oc * (H_out * W_out) + 
                                 oh * W_out + ow;
                    float grad_val = go_ptr[go_idx];

                    // --- A. Tính Gradient cho Bias (db) ---
                    // Bias cộng dồn gradient trên toàn bộ ảnh output và batch
                    gb_ptr[oc] += grad_val;

                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < K_size; ++kh) {
                            for (int kw = 0; kw < K_size; ++kw) {
                                
                                // Tính lại vị trí tương ứng trên Input
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                // Kiểm tra biên (Padding check)
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    
                                    // Index Input: [n, ic, ih, iw]
                                    int in_idx = n * (C_in * H_in * W_in) + 
                                                 ic * (H_in * W_in) + 
                                                 ih * W_in + iw;
                                    
                                    // Index Weight: [oc, ic, kh, kw]
                                    int w_idx = oc * (C_in * K_size * K_size) + 
                                                ic * (K_size * K_size) + 
                                                kh * K_size + kw;

                                    // --- B. Tính Gradient cho Weight (dW) ---
                                    // dW += Input * Grad_Output
                                    gk_ptr[w_idx] += in_ptr[in_idx] * grad_val;

                                    // --- C. Tính Gradient cho Input (dX) ---
                                    // dX += Weight * Grad_Output
                                    // Đây là bước truyền lỗi ngược về layer trước
                                    gi_ptr[in_idx] += w_ptr[w_idx] * grad_val;
                                }
                            }
                        }
                    } // End loops kernel
                }
            }
        }
    }
}

// ================= ReLU =================
// Forward: y = max(0, x)
void cpu_relu_forward(const Tensor& in, Tensor& out) {
    size_t n = in.numel();
    const float* i_ptr = (const float*)in.data_ptr();
    float* o_ptr = (float*)out.data_ptr();
    
    for(size_t i=0; i<n; ++i) {
        o_ptr[i] = (i_ptr[i] > 0.0f) ? i_ptr[i] : 0.0f;
    }
}

// Backward: grad_in = grad_out * (1 nếu x > 0 else 0)
void cpu_relu_backward(const Tensor& in, const Tensor& grad_out, Tensor& grad_in) {
    size_t n = in.numel();
    const float* i_ptr = (const float*)in.data_ptr();
    const float* go_ptr = (const float*)grad_out.data_ptr();
    float* gi_ptr = (float*)grad_in.data_ptr();
    
    for(size_t i=0; i<n; ++i) {
        gi_ptr[i] = (i_ptr[i] > 0.0f) ? go_ptr[i] : 0.0f;
    }
}

// ================= MaxPool2D =================
// Forward: Chọn giá trị lớn nhất trong cửa sổ kernel
void cpu_maxpool2d_forward(const Tensor& in, Tensor& out, Tensor& indices, int k, int s) {
    int N = in.sizes[0]; int C = in.sizes[1]; int H = in.sizes[2]; int W = in.sizes[3];
    int H_out = out.sizes[2]; int W_out = out.sizes[3];

    const float* in_ptr = (const float*)in.data_ptr();
    float* out_ptr = (float*)out.data_ptr();
    float* idx_ptr = (float*)indices.data_ptr(); // Lưu index phẳng của vị trí max

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    
                    int h_start = oh * s;
                    int w_start = ow * s;
                    int h_end = std::min(h_start + k, H);
                    int w_end = std::min(w_start + k, W);

                    float max_val = -1e20f;
                    int max_idx = -1;

                    for(int x=h_start; x<h_end; ++x) {
                        for(int y=w_start; y<w_end; ++y) {
                            int curr_idx = n*C*H*W + c*H*W + x*W + y;
                            float val = in_ptr[curr_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = curr_idx;
                            }
                        }
                    }
                    
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow;
                    out_ptr[out_idx] = max_val;
                    idx_ptr[out_idx] = (float)max_idx; // Hack: Lưu int dưới dạng float
                }
            }
        }
    }
}

// Backward: Chỉ truyền gradient về đúng vị trí max đã chọn lúc forward (Sparse update)
void cpu_maxpool2d_backward(const Tensor& grad_out, const Tensor& indices, Tensor& grad_in) {
    // Reset grad_in về 0
    std::memset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float));

    float* gi_ptr = (float*)grad_in.data_ptr();
    const float* go_ptr = (const float*)grad_out.data_ptr();
    const float* idx_ptr = (const float*)indices.data_ptr();
    
    size_t num_output = grad_out.numel();

    for(size_t i=0; i<num_output; ++i) {
        int max_idx = (int)idx_ptr[i]; // Lấy lại vị trí max
        // Cộng dồn gradient vào vị trí đó
        if (max_idx >= 0 && max_idx < (int)grad_in.numel()) {
            gi_ptr[max_idx] += go_ptr[i];
        }
    }
}

// ================= Upsample (Nearest) =================
// Forward: Phóng to ảnh bằng cách copy giá trị pixel
void cpu_upsample2d_forward(const Tensor& in, Tensor& out, int scale) {
    int N = in.sizes[0]; int C = in.sizes[1]; int H = in.sizes[2]; int W = in.sizes[3];
    int H_out = out.sizes[2]; int W_out = out.sizes[3];

    const float* in_ptr = (const float*)in.data_ptr();
    float* out_ptr = (float*)out.data_ptr();

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    // Tìm pixel gốc tương ứng (Nearest Neighbor)
                    int ih = oh / scale;
                    int iw = ow / scale;
                    
                    int in_idx = n*C*H*W + c*H*W + ih*W + iw;
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow;
                    
                    out_ptr[out_idx] = in_ptr[in_idx];
                }
            }
        }
    }
}

// Backward: Cộng dồn gradient từ vùng phóng to về pixel gốc
void cpu_upsample2d_backward(const Tensor& grad_out, Tensor& grad_in, int scale) {
    // Reset grad_in
    std::memset(grad_in.data_ptr(), 0, grad_in.numel() * sizeof(float));

    int N = grad_in.sizes[0]; int C = grad_in.sizes[1]; int H = grad_in.sizes[2]; int W = grad_in.sizes[3];
    int H_out = grad_out.sizes[2]; int W_out = grad_out.sizes[3];

    const float* go_ptr = (const float*)grad_out.data_ptr();
    float* gi_ptr = (float*)grad_in.data_ptr();

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    int ih = oh / scale;
                    int iw = ow / scale;

                    int in_idx = n*C*H*W + c*H*W + ih*W + iw;
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow;

                    gi_ptr[in_idx] += go_ptr[out_idx]; 
                }
            }
        }
    }
}

// ================= MSE LOSS =================
float cpu_mse_loss(const Tensor& pred, const Tensor& target) {
    size_t n = pred.numel();
    const float* p_ptr = (const float*)pred.data_ptr();
    const float* t_ptr = (const float*)target.data_ptr();
    
    float sum_sq = 0.0f;
    for(size_t i=0; i<n; ++i) {
        float diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
    }
    return sum_sq / n;
}

void cpu_mse_backward(const Tensor& pred, const Tensor& target, Tensor& grad_input) {
    size_t n = pred.numel();
    const float* p_ptr = (const float*)pred.data_ptr();
    const float* t_ptr = (const float*)target.data_ptr();
    float* g_ptr = (float*)grad_input.data_ptr();
    
    // Gradient công thức: dL/dx = 2/N * (x - y)
    float scale = 2.0f / n;
    
    for(size_t i=0; i<n; ++i) {
        g_ptr[i] = scale * (p_ptr[i] - t_ptr[i]);
    }
}

// ================= SGD OPTIMIZER =================
void cpu_sgd_update(Tensor& param, const Tensor& grad, float lr) {
    size_t n = param.numel();
    float* w_ptr = (float*)param.data_ptr();
    const float* g_ptr = (const float*)grad.data_ptr();
    
    for(size_t i=0; i<n; ++i) {
        w_ptr[i] -= lr * g_ptr[i];
    }
}