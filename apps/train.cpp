#include <iostream>
#include <chrono>
#include "../include/core/Tensor.h"
#include "../include/models/Autoencoder.h"
#include "../include/loss/MSELoss.h"
#include "../include/optim/SGD.h"
#include "../include/dataloader/CifarDataLoader.h"
#include "../include/utils/Serializer.h"

int main(int argc, char** argv) {
    // Config
    std::string data_path = "../data/cifar-10-batches-bin"; 
    int batch_size = 32;
    int epochs = 10;
    float lr = 0.01f;
    DeviceType device = DeviceType::CPU; 
    
    // Check GPU arg
    if (argc > 1 && std::string(argv[1]) == "gpu") {
        device = DeviceType::CUDA;
        std::cout << ">> Training on GPU" << std::endl;
    }

    // 1. Init Components
    std::cout << "Loading dataset..." << std::endl;
    CifarDataLoader loader(data_path, CifarDataLoader::Split::Train, batch_size);
    
    Autoencoder model;
    // Chuyển model sang GPU nếu cần
    if (device == DeviceType::CUDA) {
        // model.to(DeviceType::CUDA); // Bạn cần viết hàm này trong class Autoencoder
    }

    MSELoss criterion;
    SGD optimizer(model.parameters(), lr);

    // 2. Training Loop
    std::cout << "Start Training..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        loader.startEpoch(true); // Shuffle
        float total_loss = 0.0f;
        int steps = 0;

        auto t_start = std::chrono::high_resolution_clock::now();

        while (loader.hasNext()) {
            // A. Get Batch
            Batch batch = loader.nextBatch();
            Tensor images = batch.images; // [N, 3, 32, 32]
            
            // Chuyển data lên GPU nếu cần
            if (device == DeviceType::CUDA) {
                images = images.to(DeviceType::CUDA);
            }

            // B. Step
            optimizer.zero_grad();
            
            Tensor recon = model.forward(images);
            float loss = criterion.forward(recon, images); // Autoencoder: Target = Input
            
            Tensor d_loss = criterion.backward();
            model.backward(d_loss);
            
            optimizer.step();

            total_loss += loss;
            steps++;

            if (steps % 2 == 0) {
                std::cout << "\rEpoch " << epoch+1 << " Step " << steps 
                          << " Loss: " << loss << std::flush;
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\nEpoch [" << epoch+1 << "/" << epochs << "] "
                  << "Avg Loss: " << total_loss/steps 
                  << " Time: " << elapsed << "s" << std::endl;
        
        // Save Checkpoint
        Serializer::save_model(model, "weights/checkpoint_epoch_" + std::to_string(epoch+1) + ".bin");
    }

    return 0;
}   