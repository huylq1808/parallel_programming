#include <iostream>
#include <fstream>
#include <filesystem>

#include "core/Tensor.h"
#include "core/CheckError.h"
#include "dataloader/CifarDataLoader.h"
#include "utils/Serializer.h"
#include "models/Autoencoder.h"

#ifndef USE_CUDA
#error "This file requires CUDA to be enabled in CMake!"
#endif

void save_visualization_data(const Tensor& inputs, const Tensor& outputs, 
                              const std::string& filepath) {
    Tensor inputs_cpu = inputs.to(DeviceType::CPU);
    Tensor outputs_cpu = outputs.to(DeviceType::CPU);
    
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    int N = inputs_cpu.sizes[0];
    out.write((char*)&N, sizeof(int));
    
    size_t img_bytes = N * 3 * 32 * 32 * sizeof(float);
    out.write((char*)inputs_cpu.data_ptr(), img_bytes);
    
    out.write((char*)outputs_cpu.data_ptr(), img_bytes);
    
    std::cout << "Saved " << N << " images to " << filepath << std::endl;
}

int main(int argc, char* argv[]) {
    // Configuration
    std::string data_path = "../data/cifar-10-batches-bin";
    std::string weights_path = "weights/gpu_epoch_20.bin";  // Change to your best epoch
    std::string output_path = "vis_data.bin";
    int num_images = 10;  // Number of images to visualize
    
    // Parse command line arguments
    if (argc >= 2) weights_path = argv[1];
    if (argc >= 3) output_path = argv[2];
    if (argc >= 4) num_images = std::atoi(argv[3]);
    
    std::cout << "=== Autoencoder Visualization ===" << std::endl;
    std::cout << "Weights: " << weights_path << std::endl;
    std::cout << "Output:  " << output_path << std::endl;
    std::cout << "Images:  " << num_images << std::endl;
    
    // Check paths
    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Error: Data path not found: " << data_path << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(weights_path)) {
        std::cerr << "Error: Weights file not found: " << weights_path << std::endl;
        return -1;
    }
    
    // Load data
    CifarDataLoader testLoader(data_path, CifarDataLoader::Split::Test, num_images, num_images);
    
    // Create and load model
    Autoencoder model;
    model.to(DeviceType::CUDA);
    
    std::cout << "Loading model weights..." << std::endl;
    Serializer::load_model(model, weights_path);
    
    // Get a batch of images
    testLoader.startEpoch(false);  // No shuffle
    Batch batch = testLoader.nextBatch();
    
    // Move to GPU and run inference
    Tensor images_gpu = batch.images.to(DeviceType::CUDA);
    
    std::cout << "Running inference..." << std::endl;
    Tensor reconstructed = model.forward(images_gpu);
    
    // Save to binary file
    save_visualization_data(images_gpu, reconstructed, output_path);
    
    std::cout << "Done! Run the Python script to visualize." << std::endl;
    
    return 0;
}