#include "layers/MaxPool2D.h"
#include <cstring>
#include <algorithm>
#include <cfloat> // for FLT_MIN or -1e20

MaxPool2D::MaxPool2D(int k, int s) : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    input_shape_cache = input.sizes;
    int N = input.sizes[0]; int C = input.sizes[1]; 
    int H = input.sizes[2]; int W = input.sizes[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    if (out_cache.sizes != std::vector<int64_t>{N, C, H_out, W_out}) {
        out_cache = Tensor::empty({N, C, H_out, W_out});
    }
    if (indices_cache.sizes != out_cache.sizes) {
        indices_cache = Tensor::empty(out_cache.sizes);
    }

    const float* in_ptr = (const float*)input.data_ptr();
    float* out_ptr = (float*)out_cache.data_ptr();
    float* idx_ptr = (float*)indices_cache.data_ptr();

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    
                    int h_start = oh * stride;
                    int w_start = ow * stride;
                    int h_end = std::min(h_start + kernel_size, H);
                    int w_end = std::min(w_start + kernel_size, W);

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
                    idx_ptr[out_idx] = (float)max_idx; 
                }
            }
        }
    }
    return out_cache;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor dX = Tensor::zeros(input_shape_cache); // Đã memset 0

    float* gi_ptr = (float*)dX.data_ptr();
    const float* go_ptr = (const float*)grad_output.data_ptr();
    const float* idx_ptr = (const float*)indices_cache.data_ptr();
    
    size_t num_output = grad_output.numel();

    for(size_t i=0; i<num_output; ++i) {
        int max_idx = (int)idx_ptr[i];
        if (max_idx >= 0 && max_idx < (int)dX.numel()) {
            gi_ptr[max_idx] += go_ptr[i];
        }
    }
    return dX;
}