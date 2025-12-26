#include "layers/Upsample.h"
#include <cstring>

Upsample::Upsample(int scale) : scale_factor(scale) {}

Tensor Upsample::forward(const Tensor& input) {
    input_shape_cache = input.sizes;
    int N = input.sizes[0]; int C = input.sizes[1]; 
    int H = input.sizes[2]; int W = input.sizes[3];
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;

    if (out_cache.sizes != std::vector<int64_t>{N, C, H_out, W_out}) {
        out_cache = Tensor::empty({N, C, H_out, W_out});
    }

    const float* in_ptr = (const float*)input.data_ptr();
    float* out_ptr = (float*)out_cache.data_ptr();

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    // Nearest Neighbor Interpolation
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    
                    int in_idx = n*C*H*W + c*H*W + ih*W + iw;
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow;
                    
                    out_ptr[out_idx] = in_ptr[in_idx];
                }
            }
        }
    }
    return out_cache;
}

Tensor Upsample::backward(const Tensor& grad_output) {
    Tensor dX = Tensor::zeros(input_shape_cache); // Memset 0

    int N = dX.sizes[0]; int C = dX.sizes[1]; int H = dX.sizes[2]; int W = dX.sizes[3];
    int H_out = grad_output.sizes[2]; int W_out = grad_output.sizes[3];

    const float* go_ptr = (const float*)grad_output.data_ptr();
    float* gi_ptr = (float*)dX.data_ptr();

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<H_out; ++oh) {
                for(int ow=0; ow<W_out; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;

                    int in_idx = n*C*H*W + c*H*W + ih*W + iw;
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow;

                    gi_ptr[in_idx] += go_ptr[out_idx]; 
                }
            }
        }
    }
    return dX;
}