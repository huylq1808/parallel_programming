#include "../../include/layers/MaxPool2D.h"
#include "../../include/core/Ops.h"

MaxPool2D::MaxPool2D(int k, int s) : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    int N = input.sizes[0]; int C = input.sizes[1]; int H = input.sizes[2]; int W = input.sizes[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    Tensor out = Tensor::zeros({N, C, H_out, W_out}, input.device);
    // Indices tensor cùng size với output, dùng để lưu vị trí
    indices_cache = Tensor::zeros({N, C, H_out, W_out}, input.device);

    if (input.device == DeviceType::CPU) {
        cpu_maxpool2d_forward(input, out, indices_cache, kernel_size, stride);
    } 
    #ifdef USE_CUDA
    else {
        cuda_maxpool2d_forward_dispatch(input, out, indices_cache, kernel_size, stride);
    }
    #endif
    return out;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    // Input size không được lưu trực tiếp trong layer, nhưng có thể suy ra từ indices hoặc lưu input_cache nếu cần.
    // Ở đây ta tính lại size gốc hơi phức tạp chút, hoặc đơn giản là lưu input_shape lúc forward.
    // Giả sử ta biết cách map ngược lại hoặc lưu input_cache (tôi sẽ thêm input_shape vào class để tiện).
    
    // Tạm thời tính ước lượng kích thước input từ output (Hơi rủi ro nếu padding không đều)
    // Cách an toàn nhất: Lưu input size trong forward
    int N = indices_cache.sizes[0];
    int C = indices_cache.sizes[1];
    int H_out = indices_cache.sizes[2];
    int W_out = indices_cache.sizes[3];
    
    int H_in = (H_out - 1) * stride + kernel_size; // Công thức đảo ngược cơ bản
    int W_in = (W_out - 1) * stride + kernel_size;

    Tensor dX = Tensor::zeros({N, C, H_in, W_in}, indices_cache.device);

    if (indices_cache.device == DeviceType::CPU) {
        cpu_maxpool2d_backward(grad_output, indices_cache, dX);
    } 
    #ifdef USE_CUDA
    else {
        cuda_maxpool2d_backward_dispatch(grad_output, indices_cache, dX);
    }
    #endif
    return dX;
}