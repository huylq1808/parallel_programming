#include <iostream>
#include "../include/core/Tensor.h"
#include "../include/models/Autoencoder.h"
#include "../include/utils/Serializer.h"
#include "../include/dataloader/CifarDataLoader.h"

int main() {
    // 1. Load Model
    Autoencoder model;
    try {
        Serializer::load_model(model, "weights/checkpoint_epoch_10.bin");
    } catch (...) {
        std::cerr << "Warning: No checkpoint found, using random weights." << std::endl;
    }

    // 2. Load Test Data
    CifarDataLoader test_loader("data/cifar-10-batches-bin", CifarDataLoader::Split::Test, 1);
    test_loader.startEpoch(false);
    
    // Lấy 1 ảnh
    Batch b = test_loader.nextBatch();
    Tensor img = b.images; // [1, 3, 32, 32]

    // 3. Inference
    Tensor output = model.forward(img);

    std::cout << "Inference Done." << std::endl;
    std::cout << "Input Shape: " << img.sizes[2] << "x" << img.sizes[3] << std::endl;
    std::cout << "Output Shape: " << output.sizes[2] << "x" << output.sizes[3] << std::endl;

    // TODO: Bạn có thể viết thêm hàm lưu Tensor ra file ảnh .png để xem kết quả
    return 0;
}