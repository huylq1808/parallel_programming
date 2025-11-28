#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>

struct Shape {
    int n = 0; // batch size
    int c = 0; // channels
    int h = 0; // height
    int w = 0; // width
};

struct Tensor
{
    std::vector<float> data;
    int n = 0, c = 0, h = 0, w = 0;

    Tensor() = default;

    Tensor(int n_, int c_, int h_, int w_) {
        resize(n_, c_, h_, w_);
    }

    void resize(int n_, int c_, int h_, int w_) {
        n = n_, c = c_, h = h_, w = w_;
        data.assign(static_cast<size_t>(n) * c * h * w, 0.0f); 
    }

    size_t elements() const {
        return data.size();
    }

    // Layout: NCHW
    // https://uxlfoundation.github.io/oneDNN/dev_guide_understanding_memory_formats.html
    size_t index(int nIdx, int cIdx, int hIdx, int wIdx) const {
        return (((static_cast<size_t>(nIdx) * c + cIdx) * h + hIdx) * w + wIdx);
    }

    float &operator()(int nIdx, int cIdx, int hIdx, int wIdx) {
        return data[index(nIdx, cIdx, hIdx, wIdx)];
    }

    const float &operator()(int nIdx, int cIdx, int hIdx, int wIdx) const {
        return data[index(nIdx, cIdx, hIdx, wIdx)];
    }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }
};

struct Batch {
    Tensor images;
    std::vector<uint8_t> labels;
};

class CifarDataLoader {
public:
    enum class Split {
        Train,
        Test
    };

    CifarDataLoader(const std::filesystem::path &root, Split split) {
        loadSplit(root, split);

        // Shuffle sample
        permutation_.resize(numSamples_);
        std::iota(permutation_.begin(), permutation_.end(), 0);
    }

    size_t numSamples() const {
        return numSamples_;
    }

    void startEpoch(bool shuffle = true) {
        cursor_ = 0;
        if (shuffle) {
            std::shuffle(permutation_.begin(), permutation_.end(), rng_);
        }
    }

    Batch nextBatch(int batchSize) {
        if (numSamples_ == 0) 
            throw std::runtime_error("CifarDataLoader: dataset is empty");
    
        if (cursor_ >= numSamples_)
            cursor_ = 0;
        
        // actualBatch có thể nhỏ hơn batchSize nếu còn ít sample 
        const int actualBatch = std::min<int>(batchSize, static_cast<int>(numSamples_ - cursor_));
        Batch batch;
        batch.images.resize(actualBatch, channels_, height_, width_);
        batch.labels.resize(actualBatch);

        const size_t imageSize = static_cast<size_t>(channels_) * height_ * width_;
        for (int i = 0; i < actualBatch; i++, cursor_++) {
            size_t sampleIdx = permutation_[cursor_];
            const float *src = &storage_[sampleIdx * imageSize];
            float *dst = &batch.images.data[static_cast<size_t>(i) * imageSize];
            std::copy(src, src + imageSize, dst);
            batch.labels[i] = labels_[sampleIdx];
        }

        return batch;
    }

private:
    static constexpr int channels_ = 3;
    static constexpr int height_ = 32;
    static constexpr int width_ = 32;
    static constexpr size_t imageBytes_ = static_cast<size_t>(channels_) * height_ * width_;

    size_t numSamples_ = 0;
    std::vector<float> storage_;
    std::vector<uint8_t> labels_;
    std::vector<size_t> permutation_;
    size_t cursor_ = 0;
    std::mt19937 rng_;

    void loadSplit(const std::filesystem::path &root, Split split) {
        std::vector<std::filesystem::path> files;
        if (split == Split::Train) {
            for (int i = 1; i <= 5; i++) {
                files.emplace_back(root / ("data_batch_" + std::to_string(i) + ".bin"));
            }
        }
        else 
            files.emplace_back(root / "test_batch.bin");

        for (auto &file : files)
            readFile(file);

        numSamples_ = labels_.size();
    }

    void readFile(const std::filesystem::path &file) {
        std::ifstream input(file, std::ios::binary);
        if (!input)
            throw std::runtime_error("Cannot open " + file.string());

        while (true) {
            uint8_t label = 0;
            input.read(reinterpret_cast<char *>(&label), 1);
            if (!input)
                break;

            std::array<uint8_t, imageBytes_> buffer{};
            input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
            if (!input)
                throw std::runtime_error("Unexpected EOF in " + file.string());

            labels_.push_back(label);
            for (size_t i = 0; i < buffer.size(); i++)
                storage_.push_back(static_cast<float>(buffer[i]) / 255.0f);
        }
    }
};

class ReLU {
private: 
    std::vector<size_t> mask_;
public: 
    Tensor forward(Tensor &input) {
        mask_.assign(input.elements(), 0);
        Tensor output = input; 
        for (size_t i = 0; i < input.elements(); i++) {
            if (input.data[i] > 0) 
                mask_[i] = 1;
            else 
                output.data[i] = 0;
        }
        return output;
    }

    Tensor backward(Tensor &output) {
        Tensor input = output;
        for (size_t i = 0; i < output.elements(); i++) {
            input.data[i] *= static_cast<float>(mask_[i]);
        }
        return input;
    }
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cifar_root> [epochs] [batch_size] [learning_rate]\n";
        return 1;
    }

    const std::filesystem::path cifarRoot = argv[1];
    if (!std::filesystem::exists(cifarRoot)) {
        std::cerr << "Dataset folder not found: " << cifarRoot << "\n";
        return 1;
    }

    return 0;
}