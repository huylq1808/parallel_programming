#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>

#include "../include/core/Tensor.h"
#include "../include/core/CheckError.h" // Để dùng CHECK macro nếu cần
#include "../include/models/Autoencoder.h"
#include "../include/loss/MSELoss.h"
#include "../include/optim/SGD.h"
#include "../include/dataloader/CifarDataLoader.h"
#include "../include/utils/Serializer.h"

#ifndef USE_CUDA
#error "This file requires CUDA to be enabled in CMake!"
#endif

// Hàm Validation GPU
float validate(Autoencoder& model, CifarDataLoader& loader, MSELoss& criterion) {
    loader.startEpoch(false);
    float total_loss = 0.0f;
    int steps = 0;

    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        // Chuyển data sang GPU
        Tensor images_gpu = batch.images.to(DeviceType::CUDA);

        Tensor recon = model.forward(images_gpu);
        float loss = criterion.forward(recon, images_gpu);
        
        total_loss += loss;
        steps++;
    }
    return (steps > 0) ? total_loss / steps : 0.0f;
}

int main() {
    // Config
    std::string data_path = "../data/cifar-10-batches-bin";
    int batch_size = 64;
    int epochs = 10;
    float lr = 0.001f;
    int train_samples = 40000;
    int val_samples = 10000;

    std::cout << ">> Mode: GPU Training (CUDA)" << std::endl;

    if (!std::filesystem::exists(data_path)) return -1;

    CifarDataLoader trainLoader(data_path, CifarDataLoader::Split::Train, batch_size, train_samples);
    CifarDataLoader valLoader(data_path, CifarDataLoader::Split::Test, batch_size, val_samples);

    Autoencoder model;
    // 1. Chuyển Model sang GPU ngay lập tức
    model.to(DeviceType::CUDA); 

    MSELoss criterion;
    // 2. Init Optimizer SAU KHI chuyển model sang GPU (để nó trỏ vào weights GPU)
    SGD optimizer(model.parameters(), lr);

    std::cout << "\n========== START TRAINING (GPU) ==========" << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        trainLoader.startEpoch(true);
        float train_loss = 0.0f;
        int train_steps = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();

        while (trainLoader.hasNext()) {
            Batch batch = trainLoader.nextBatch();
            
            // 3. Move Batch to GPU
            Tensor images_gpu = batch.images.to(DeviceType::CUDA);

            optimizer.zero_grad();
            
            Tensor recon = model.forward(images_gpu);
            float loss = criterion.forward(recon, images_gpu);
            
            Tensor d_loss = criterion.backward();
            model.backward(d_loss);
            optimizer.step();

            train_loss += loss;
            train_steps++;

            if (train_steps % 100 == 0){
                std::cout << "\rEpoch " << epoch+1 << " | Step " << train_steps 
                        << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::flush;
            }
        }
        
        float avg_val_loss = validate(model, valLoader, criterion);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "\rEpoch [" << epoch+1 << "/" << epochs << "] "
                  << "Time: " << std::chrono::duration<double>(t_end - t_start).count() << "s | "
                  << "Train Loss: " << (train_loss/train_steps) << " | "
                  << "Val Loss: " << avg_val_loss << std::endl;

        std::filesystem::create_directory("weights");
        Serializer::save_model(model, "weights/gpu_epoch_" + std::to_string(epoch+1) + ".bin");
    }
    return 0;
}