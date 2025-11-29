#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring> // Cần cho std::memcpy

// --- CÁC FILE INCLUDE CẦN THIẾT ---
#include "../core/Tensor.h"
#include "../core/CheckError.h" // [QUAN TRỌNG] Để dùng macro CHECK
#include "../models/Autoencoder.h"

#ifdef USE_CUDA
#include <cuda_runtime.h> // [QUAN TRỌNG] Để dùng cudaMemcpy
#endif

class Serializer {
public:
    // Lưu model ra file .bin
    static void save_model(Autoencoder& model, const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing: " + filepath);

        auto params = model.parameters();
        
        // 1. Lưu số lượng tensor
        size_t num_params = params.size();
        out.write((char*)&num_params, sizeof(size_t));

        for (auto* t : params) {
            // 2. Lưu số chiều (dims)
            size_t dims = t->sizes.size();
            out.write((char*)&dims, sizeof(size_t));

            // 3. Lưu sizes
            out.write((char*)t->sizes.data(), dims * sizeof(int64_t));

            // 4. Lưu data (luôn chuyển về CPU để lưu)
            Tensor cpu_t = t->to(DeviceType::CPU);
            out.write((char*)cpu_t.data_ptr(), cpu_t.numel() * sizeof(float));
        }
        std::cout << "Model saved to " << filepath << std::endl;
    }

    // Load model từ file .bin
    static void load_model(Autoencoder& model, const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading: " + filepath);

        auto params = model.parameters();
        size_t num_params_file;
        in.read((char*)&num_params_file, sizeof(size_t));

        if (num_params_file != params.size()) {
            throw std::runtime_error("Model architecture mismatch! File has " + 
                std::to_string(num_params_file) + " params, Model has " + 
                std::to_string(params.size()));
        }

        for (auto* t : params) {
            size_t dims;
            in.read((char*)&dims, sizeof(size_t));

            std::vector<int64_t> file_sizes(dims);
            in.read((char*)file_sizes.data(), dims * sizeof(int64_t));

            // Check size consistency
            if (file_sizes != t->sizes) throw std::runtime_error("Tensor shape mismatch!");

            // Đọc data vào buffer tạm
            std::vector<float> buffer(t->numel());
            in.read((char*)buffer.data(), buffer.size() * sizeof(float));

            // Copy vào Tensor (xử lý device)
            if (t->device == DeviceType::CPU) {
                std::memcpy(t->data_ptr(), buffer.data(), buffer.size() * sizeof(float));
            } 
            #ifdef USE_CUDA
            else {
                // Bây giờ trình biên dịch đã hiểu CHECK và cudaMemcpy
                CHECK(cudaMemcpy(t->data_ptr(), buffer.data(), 
                                 buffer.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
            #endif
        }
        std::cout << "Model loaded from " << filepath << std::endl;
    }
};