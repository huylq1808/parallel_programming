#include <iostream>
#include <chrono>
#include <iomanip>

// Include Core Headers
#include "core/Tensor.h"
#include "core/CheckError.h"
#include "models/Autoencoder.h"
#include "loss/MSELoss.h"
#include "optim/SGD.h"
#include "dataloader/CifarDataLoader.h"
#include "utils/Serializer.h"

// Hàm đánh giá (Validation)
float validate(Autoencoder& model, CifarDataLoader& loader, MSELoss& criterion, DeviceType device) {
    loader.startEpoch(false); // Không cần shuffle khi validate
    float total_loss = 0.0f;
    int steps = 0;

    while (loader.hasNext()) {
        Batch batch = loader.nextBatch();
        Tensor images = batch.images;
        
        if (device == DeviceType::CUDA) {
            images = images.to(DeviceType::CUDA);
        }

        // Forward (Không cần tính gradient, nhưng kiến trúc hiện tại chưa có mode eval, nên cứ chạy forward)
        Tensor recon = model.forward(images);
        float loss = criterion.forward(recon, images);
        
        total_loss += loss;
        steps++;
    }
    return (steps > 0) ? total_loss / steps : 0.0f;
}

int main(int argc, char** argv) {
    // 0. Config
    std::string data_path = "../data/cifar-10-batches-bin"; 
    int batch_size = 32;
    int epochs = 5;
    float lr = 0.001f;
    int train_samples = 1000; // [YÊU CẦU] Chỉ lấy 1k mẫu train
    int val_samples = 100;    // Lấy 100 mẫu để test
    
    DeviceType device = DeviceType::CPU; 
    
    if (argc > 1 && std::string(argv[1]) == "gpu") {
#ifdef USE_CUDA
        device = DeviceType::CUDA;
        std::cout << ">> Mode: GPU Training" << std::endl;
#else
        std::cout << ">> Warning: CUDA not enabled. Fallback to CPU." << std::endl;
#endif
    } else {
        std::cout << ">> Mode: CPU Training" << std::endl;
    }

    // 1. Init DataLoaders
    // Check path
    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Error: Data path not found: " << data_path << std::endl;
        return -1;
    }

    std::cout << "Loading Train Set (Subset 1000)..." << std::endl;
    CifarDataLoader trainLoader(data_path, CifarDataLoader::Split::Train, batch_size, train_samples);
    
    std::cout << "Loading Test Set (Subset 100)..." << std::endl;
    CifarDataLoader valLoader(data_path, CifarDataLoader::Split::Test, batch_size, val_samples);

    // 2. Init Model & Optim
    Autoencoder model;
    
    // Chuyển model sang GPU
    if (device == DeviceType::CUDA) {
        model.to_gpu(); // Hàm này đã có trong Autoencoder.h bạn cung cấp trước đó
    }

    MSELoss criterion;
    SGD optimizer(model.parameters(), lr);

    // 3. Training Loop
    std::cout << "\n========== START TRAINING ==========" << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        // --- TRAINING PHASE ---
        trainLoader.startEpoch(true);
        float train_loss = 0.0f;
        int train_steps = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();

        while (trainLoader.hasNext()) {
            Batch batch = trainLoader.nextBatch();
            Tensor images = batch.images; // [N, 3, 32, 32]
            
            if (device == DeviceType::CUDA) {
                images = images.to(DeviceType::CUDA);
            }

            // Zero Grad
            //std::cout << "Step: " << train_steps << " - Zeroing gradients.\n";
            optimizer.zero_grad();
            
            // Forward
            //std::cout << "Step: " << train_steps << " - Forward pass.\n";
            Tensor recon = model.forward(images);
            float loss = criterion.forward(recon, images);
            
            // Backward
            //std::cout << "Step: " << train_steps << " - Backward pass.\n";
            Tensor d_loss = criterion.backward(); // dL/dX
            model.backward(d_loss);
            
            // Update
            //std::cout << "Step: " << train_steps << " - Optimizer step.\n";
            optimizer.step();

            train_loss += loss;
            train_steps++;

            // Log mỗi 10 bước
            std::cout << "\rEpoch [" << epoch+1 << "/" << epochs << "] "
                        << "Step: " << train_steps << " | "
                        << "Loss: " << std::fixed << std::setprecision(6) << (train_loss / train_steps) 
                        << std::flush;
            
        }
        
        float avg_train_loss = train_loss / train_steps;

        // --- VALIDATION PHASE ---
        float avg_val_loss = validate(model, valLoader, criterion, device);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        // End Epoch Log
        std::cout << "\rEpoch [" << epoch+1 << "/" << epochs << "] "
                  << "Time: " << std::setw(4) << (int)elapsed << "s | "
                  << "Train Loss: " << avg_train_loss << " | "
                  << "Val Loss: " << avg_val_loss << std::endl;
        
        // Save Checkpoint
        std::string save_path = "weights/epoch_" + std::to_string(epoch+1) + ".bin";
        
        // Đảm bảo thư mục weights tồn tại
        std::filesystem::create_directory("weights");
        Serializer::save_model(model, save_path);
    }

    std::cout << "Done." << std::endl;
    return 0;
}