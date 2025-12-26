#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>

#include "core/Tensor.h"
#include "models/Autoencoder.h"
#include "loss/MSELoss.h"
#include "optim/SGD.h"
#include "dataloader/CifarDataLoader.h"
#include "utils/Serializer.h"

// Hàm Validation CPU
float validate(Autoencoder& model, CifarDataLoader& loader, MSELoss& criterion) {
    loader.startEpoch(false);
    float total_loss = 0.0f;
    int steps = 0;

    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        // Forward trực tiếp trên CPU
        Tensor recon = model.forward(batch.images);
        float loss = criterion.forward(recon, batch.images);
        total_loss += loss;
        steps++;
    }
    return (steps > 0) ? total_loss / steps : 0.0f;
}

int main(int argc, char** argv) {
    // Config
    std::string data_path = "../data/cifar-10-batches-bin";
    int batch_size = 32;
    int epochs = 5;
    float lr = 0.001f;
    int train_samples = 100; 
    int val_samples = 100;

    if (argc > 1) data_path = argv[1];
    if (argc > 2) batch_size = std::atoi(argv[2]);
    if (argc > 3) epochs = std::atoi(argv[3]);
    if (argc > 4) lr = std::atof(argv[4]);
    if (argc > 5) train_samples = std::atoi(argv[5]);
    if (argc > 6) val_samples = std::atoi(argv[6]);

    std::cout << ">> Mode: CPU Training ONLY" << std::endl;

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Error: Data path not found!" << std::endl;
        return -1;
    }

    CifarDataLoader trainLoader(data_path, CifarDataLoader::Split::Train, batch_size, train_samples);
    CifarDataLoader valLoader(data_path, CifarDataLoader::Split::Test, batch_size, val_samples);

    Autoencoder model; // Mặc định ở CPU
    MSELoss criterion;
    SGD optimizer(model.parameters(), lr);

    std::cout << "\n========== START TRAINING (CPU) ==========" << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        trainLoader.startEpoch(true);
        float train_loss = 0.0f;
        int train_steps = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();

        while (trainLoader.hasNext()) {
            Batch batch = trainLoader.nextBatch();
            
            optimizer.zero_grad();
            
            Tensor recon = model.forward(batch.images);
            float loss = criterion.forward(recon, batch.images);
            
            Tensor d_loss = criterion.backward();
            model.backward(d_loss);
            optimizer.step();

            train_loss += loss;
            train_steps++;

            std::cout << "\rEpoch " << epoch+1 << " | Step " << train_steps 
                    << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::flush;
    
        }
        
        float avg_val_loss = validate(model, valLoader, criterion);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "\rEpoch [" << epoch+1 << "/" << epochs << "] "
                  << "Time: " << std::chrono::duration<double>(t_end - t_start).count() << "s | "
                  << "Train Loss: " << (train_loss/train_steps) << " | "
                  << "Val Loss: " << avg_val_loss << std::endl;

        std::filesystem::create_directory("weights");
        Serializer::save_model(model, "weights/cpu_epoch_" + std::to_string(epoch+1) + ".bin");
    }
    return 0;
}