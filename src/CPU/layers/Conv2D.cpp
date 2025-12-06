#include "layers/Conv2D.h"
#include <cstring>
#include <cmath>
#include <algorithm>

Conv2D::Conv2D(int in, int out, int k, int s, int p) 
    : in_c(in), out_c(out), k_size(k), stride(s), padding(p) 
{
    // Khởi tạo weights trên CPU
    W = Tensor::randn({out, in, k, k}, 0.0f, 0.01f);
    b = Tensor::zeros({out});
    W.requires_grad = true; 
    b.requires_grad = true;
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;
    int N = input.sizes[0]; 
    int H_in = input.sizes[2]; 
    int W_in = input.sizes[3];
    
    int H_out = (H_in + 2*padding - k_size) / stride + 1;
    int W_out = (W_in + 2*padding - k_size) / stride + 1;

    // Chuẩn bị output buffer
    if (out_cache.sizes != std::vector<int64_t>{N, out_c, H_out, W_out}) {
        out_cache = Tensor::empty({N, out_c, H_out, W_out});
    }

    // --- LOGIC TÍNH TOÁN TRỰC TIẾP ---
    const float* in_ptr = (const float*)input.data_ptr();
    const float* w_ptr = (const float*)W.data_ptr();
    const float* b_ptr = (const float*)b.data_ptr();
    float* out_ptr = (float*)out_cache.data_ptr();

    // Reset memory output về 0 hoặc khởi tạo bằng bias trước khi cộng dồn
    // Ở đây ta khởi tạo bằng bias trong vòng lặp luôn cho tiện
    
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    
                    float sum = b_ptr[oc]; // Init với bias

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            for (int kw = 0; kw < k_size; ++kw) {
                                int hi = oh * stride - padding + kh;
                                int wi = ow * stride - padding + kw;

                                if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int in_idx = n * (in_c * H_in * W_in) + ic * (H_in * W_in) + hi * W_in + wi;
                                    int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + kh * k_size + kw;
                                    sum += in_ptr[in_idx] * w_ptr[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * (out_c * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow;
                    out_ptr[out_idx] = sum;
                }
            }
        }
    }
    return out_cache;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    if(!W.grad) W.ensure_grad();
    if(!b.grad) b.ensure_grad();
    
    // Cấp phát gradient cho Input
    Tensor dIn = Tensor::zeros(input_cache.sizes);
    
    // Reset gradient của Weights và Bias về 0 trước khi cộng dồn
    W.zero_grad();
    b.zero_grad();

    // --- LOGIC BACKWARD ---
    int N = input_cache.sizes[0];
    int H_in = input_cache.sizes[2];
    int W_in = input_cache.sizes[3];
    int H_out = grad_output.sizes[2];
    int W_out = grad_output.sizes[3];

    const float* in_ptr = (const float*)input_cache.data_ptr();
    const float* w_ptr = (const float*)W.data_ptr();
    const float* go_ptr = (const float*)grad_output.data_ptr();

    float* gi_ptr = (float*)dIn.data_ptr();
    float* gk_ptr = (float*)W.grad->data_ptr();
    float* gb_ptr = (float*)b.grad->data_ptr();

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    
                    int go_idx = n * (out_c * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow;
                    float grad_val = go_ptr[go_idx];

                    // Gradient Bias
                    gb_ptr[oc] += grad_val;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            for (int kw = 0; kw < k_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    int in_idx = n * (in_c * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw;
                                    int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + kh * k_size + kw;

                                    // Gradient Weight: Input * Grad_Out
                                    gk_ptr[w_idx] += in_ptr[in_idx] * grad_val;
                                    // Gradient Input: Weight * Grad_Out
                                    gi_ptr[in_idx] += w_ptr[w_idx] * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return dIn;
}

// do notthing in CPU version 
void Conv2D::to(DeviceType device) {
    return;
}