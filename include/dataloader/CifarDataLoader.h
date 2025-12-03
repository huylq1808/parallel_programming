#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <array>
#include <iostream> 
#include "core/Tensor.h"

struct Batch {
    Tensor images;
    Tensor labels;
};

class CifarDataLoader {
public:
    enum class Split { Train, Test };

    // Thêm tham số limit_samples (mặc định = 0 là lấy hết)
    CifarDataLoader(const std::string& root, Split split, int batch_size, int limit_samples = 0) 
        : batch_size_(batch_size) 
    {
        loadSplit(root, split);
        
        // Nếu có giới hạn mẫu, cắt bớt vector storage và labels
        if (limit_samples > 0 && limit_samples < numSamples_) {
            numSamples_ = limit_samples;
            labels_.resize(numSamples_);
            // Ảnh hưởng đến storage_ là phức tạp hơn vì nó là flat vector
            // Tuy nhiên để đơn giản, ta chỉ cần giới hạn permutation_ (danh sách chỉ số)
            // Storage vẫn giữ nguyên nhưng ta chỉ truy cập 1000 mẫu đầu.
        }

        // Tạo hoán vị ngẫu nhiên
        permutation_.resize(numSamples_);
        std::iota(permutation_.begin(), permutation_.end(), 0);
        rng_.seed(std::random_device{}());

        std::cout << ">> DataLoader initialized. Samples: " << numSamples_ 
                  << " | Batch Size: " << batch_size_ << std::endl;
    }

    size_t numSamples() const { return numSamples_; }

    void startEpoch(bool shuffle = true) {
        cursor_ = 0;
        if (shuffle) {
            std::shuffle(permutation_.begin(), permutation_.end(), rng_);
        }
    }

    bool hasNext() const {
        return cursor_ < numSamples_;
    }

    Batch nextBatch() {
        if (cursor_ >= numSamples_) 
            throw std::runtime_error("No more batches. Did you forget startEpoch()?");

        int currentBatchSize = std::min<int>(batch_size_, static_cast<int>(numSamples_ - cursor_));
        
        // Tạo Tensor trên CPU
        Batch batch;
        batch.images = Tensor::zeros({currentBatchSize, channels_, height_, width_}, DeviceType::CPU);
        float* img_ptr = (float*)batch.images.data_ptr();

        batch.labels = Tensor::zeros({currentBatchSize}, DeviceType::CPU);
        float* label_ptr = (float*)batch.labels.data_ptr();

        size_t imageSize = channels_ * height_ * width_;

        for (int i = 0; i < currentBatchSize; ++i) {
            size_t sampleIdx = permutation_[cursor_];
            
            // Copy Image
            const float* src = &storage_[sampleIdx * imageSize];
            float* dst = &img_ptr[i * imageSize];
            std::copy(src, src + imageSize, dst);

            // Copy Label
            label_ptr[i] = static_cast<float>(labels_[sampleIdx]);

            cursor_++;
        }

        return batch;
    }

private:
    static constexpr int channels_ = 3;
    static constexpr int height_ = 32;
    static constexpr int width_ = 32;
    static constexpr size_t imageBytes_ = channels_ * height_ * width_;

    int batch_size_;
    size_t numSamples_ = 0;
    std::vector<float> storage_;    
    std::vector<uint8_t> labels_;
    std::vector<size_t> permutation_;
    size_t cursor_ = 0;
    std::mt19937 rng_;

    void loadSplit(const std::filesystem::path& root, Split split) {
        std::vector<std::filesystem::path> files;
        if (split == Split::Train) {
            for (int i = 1; i <= 5; ++i) {
                // Kiểm tra file tồn tại để tránh crash
                auto p = root / ("data_batch_" + std::to_string(i) + ".bin");
                if(std::filesystem::exists(p)) files.emplace_back(p);
            }
        } else {
            auto p = root / "test_batch.bin";
            if(std::filesystem::exists(p)) files.emplace_back(p);
        }

        if (files.empty()) {
            std::cerr << "Warning: No data files found in " << root << std::endl;
            return; 
        }

        for (const auto& file : files) {
            readFile(file);
        }
        numSamples_ = labels_.size();
    }

    void readFile(const std::filesystem::path& file) {
        std::ifstream input(file, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open " + file.string());

        while (true) {
            uint8_t label = 0;
            if (!input.read(reinterpret_cast<char*>(&label), 1)) break;

            std::array<uint8_t, imageBytes_> buffer;
            if (!input.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) break;

            labels_.push_back(label);
            for (uint8_t b : buffer) {
                storage_.push_back(static_cast<float>(b) / 255.0f);
            }
        }
    }
};