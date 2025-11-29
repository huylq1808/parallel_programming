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
#include "../core/Tensor.h"

// Struct chứa một Batch dữ liệu
struct Batch {
    Tensor images;          // Shape: [BatchSize, 3, 32, 32]
    Tensor labels;          // Shape: [BatchSize] (hoặc vector<uint8_t> nếu muốn)
};

class CifarDataLoader {
public:
    enum class Split { Train, Test };

    CifarDataLoader(const std::string& root, Split split, int batch_size) 
        : batch_size_(batch_size) 
    {
        loadSplit(root, split);
        
        // Tạo hoán vị ngẫu nhiên cho việc shuffle
        permutation_.resize(numSamples_);
        std::iota(permutation_.begin(), permutation_.end(), 0);
        rng_.seed(std::random_device{}());
    }

    size_t numSamples() const { return numSamples_; }

    // Gọi đầu mỗi Epoch để xáo trộn dữ liệu
    void startEpoch(bool shuffle = true) {
        cursor_ = 0;
        if (shuffle) {
            std::shuffle(permutation_.begin(), permutation_.end(), rng_);
        }
    }

    // Kiểm tra còn batch nào không
    bool hasNext() const {
        return cursor_ < numSamples_;
    }

    // Lấy batch tiếp theo
    Batch nextBatch() {
        if (cursor_ >= numSamples_) 
            throw std::runtime_error("No more batches. Did you forget startEpoch()?");

        int currentBatchSize = std::min<int>(batch_size_, static_cast<int>(numSamples_ - cursor_));
        
        // 1. Tạo Tensor Images [N, C, H, W] trên CPU
        Batch batch;
        batch.images = Tensor::zeros({currentBatchSize, channels_, height_, width_}, DeviceType::CPU);
        float* img_ptr = (float*)batch.images.data_ptr();

        // 2. Tạo Tensor Labels [N] (để tính loss cho tiện)
        // Lưu ý: Tensor hiện tại chỉ hỗ trợ float, nên ta ép kiểu label sang float
        batch.labels = Tensor::zeros({currentBatchSize}, DeviceType::CPU);
        float* label_ptr = (float*)batch.labels.data_ptr();

        // 3. Copy dữ liệu từ storage vào Tensor
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
    static constexpr size_t imageBytes_ = channels_ * height_ * width_; // 3072 bytes

    int batch_size_;
    size_t numSamples_ = 0;
    std::vector<float> storage_;    // Buffer lưu toàn bộ dataset trên RAM
    std::vector<uint8_t> labels_;
    std::vector<size_t> permutation_;
    size_t cursor_ = 0;
    std::mt19937 rng_;

    void loadSplit(const std::filesystem::path& root, Split split) {
        std::vector<std::filesystem::path> files;
        if (split == Split::Train) {
            for (int i = 1; i <= 5; ++i) {
                files.emplace_back(root / ("data_batch_" + std::to_string(i) + ".bin"));
            }
        } else {
            files.emplace_back(root / "test_batch.bin");
        }

        for (const auto& file : files) {
            readFile(file);
        }
        numSamples_ = labels_.size();
        // std::cout << "Loaded " << numSamples_ << " samples from " << root << std::endl;
    }

    void readFile(const std::filesystem::path& file) {
        std::ifstream input(file, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open " + file.string());

        while (true) {
            // Định dạng CIFAR-10 binary: <1 byte label><3072 bytes image>
            uint8_t label = 0;
            if (!input.read(reinterpret_cast<char*>(&label), 1)) break;

            std::array<uint8_t, imageBytes_> buffer;
            if (!input.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) break;

            labels_.push_back(label);
            // Chuẩn hóa về [0, 1] và lưu vào storage
            for (uint8_t b : buffer) {
                storage_.push_back(static_cast<float>(b) / 255.0f);
            }
        }
    }
};