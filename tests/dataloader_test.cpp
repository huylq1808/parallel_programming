#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstring> 
#include "../include/core/Tensor.h"
#include "../include/dataloader/CifarDataLoader.h"

// Hàm vẽ ảnh bằng ký tự ASCII
void visualize_ascii(const Tensor& img, int label) {
    // img shape: [3, 32, 32]
    // Ta lấy kênh Green (kênh 1) để hiển thị độ sáng đại diện
    const float* data = (const float*)img.data_ptr();
    int H = 32;
    int W = 32;
    
    // Ký tự từ tối đến sáng
    std::string chars = " .:-=+*#%@";

    std::cout << "\n--- Visualizing Sample (Label: " << label << ") ---" << std::endl;
    std::cout << "    (ASCII Art Representation)" << std::endl;
    std::cout << " +--------------------------------+" << std::endl;
    
    for (int h = 0; h < H; ++h) {
        std::cout << " |";
        for (int w = 0; w < W; ++w) {
            // Lấy giá trị pixel trung bình của 3 kênh RGB cho chính xác hơn
            float r = data[0*H*W + h*W + w];
            float g = data[1*H*W + h*W + w];
            float b = data[2*H*W + h*W + w];
            float avg = (r + g + b) / 3.0f;

            // Map 0.0-1.0 sang index 0-9
            int char_idx = (int)(avg * (chars.size() - 1));
            if (char_idx < 0) char_idx = 0;
            if (char_idx >= chars.size()) char_idx = chars.size() - 1;
            
            std::cout << chars[char_idx];
        }
        std::cout << "|" << std::endl;
    }
    std::cout << " +--------------------------------+" << std::endl;
}

int main() {
    std::string data_path = "../data/cifar-10-batches-bin";
    int batch_size = 4;

    std::cout << "Checking Dataset path: " << data_path << std::endl;

    try {
        // 1. Init Loader
        CifarDataLoader loader(data_path, CifarDataLoader::Split::Train, batch_size);
        std::cout << "[PASS] Loader Initialized." << std::endl;
        std::cout << "Total Samples: " << loader.numSamples() << std::endl;

        // 2. Start Epoch
        loader.startEpoch(true); // Shuffle

        // 3. Get First Batch
        if (loader.hasNext()) {
            Batch b = loader.nextBatch();
            Tensor imgs = b.images;
            Tensor lbls = b.labels;

            std::cout << "[PASS] Batch Loaded." << std::endl;
            std::cout << "Images Shape: [";
            for(auto s : imgs.sizes) std::cout << s << " ";
            std::cout << "]" << std::endl; // Expect [4, 3, 32, 32]

            std::cout << "Labels Shape: [";
            for(auto s : lbls.sizes) std::cout << s << " ";
            std::cout << "]" << std::endl; // Expect [4]

            // Check Data Range (Phải từ 0.0 đến 1.0)
            float* ptr = (float*)imgs.data_ptr();
            float min_val = 1000.0f, max_val = -1000.0f;
            for(size_t i=0; i<imgs.numel(); ++i) {
                if(ptr[i] < min_val) min_val = ptr[i];
                if(ptr[i] > max_val) max_val = ptr[i];
            }
            std::cout << "Pixel Range: [" << min_val << ", " << max_val << "]" << std::endl;
            
            if (min_val >= 0.0f && max_val <= 1.0f) 
                std::cout << "[PASS] Data Normalization seems correct." << std::endl;
            else 
                std::cout << "[WARN] Data might not be normalized correctly." << std::endl;

            // 4. Visualize hình đầu tiên trong batch
            // Label mapping (CIFAR-10)
            // 0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 
            // 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
            float* l_ptr = (float*)lbls.data_ptr();
            
            // Lấy ảnh đầu tiên trong batch để vẽ
            // Slice thủ công: tạo tensor view hoặc copy data
            // Ở đây copy data cho đơn giản
            Tensor single_img = Tensor::zeros({3, 32, 32}, DeviceType::CPU);
            size_t img_size = 3*32*32;
            std::memcpy(single_img.data_ptr(), ptr, img_size * sizeof(float));
            
            visualize_ascii(single_img, (int)l_ptr[0]);
        }

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Error: " << e.what() << std::endl;
        std::cerr << "Hint: Have you downloaded 'cifar-10-binary.tar.gz' and extracted it to 'data/cifar-10-batches-bin'?" << std::endl;
        return 1;
    }

    return 0;
}