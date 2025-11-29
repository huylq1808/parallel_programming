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

class MaxPool2x2 {
private: 
    Shape inputShape_;
    std::vector<int> maxIndices_;

public: 
    Tensor forward(Tensor &input) {
        inputShape_ = { input.n, input.c, input.h, input.w };
        const int outH = input.h / 2;
        const int outW = input.w / 2;
        Tensor output(input.n, input.c, outH, outW);
        maxIndices_.resize(output.elements(), -1);

        for (int n = 0; n < input.n; n++) {
            for (int c = 0; c < input.c; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        float maxVal = -std::numeric_limits<float>::infinity();
                        int maxIdx = -1;
                        for (int kh = 0; kh < 2; kh++) {
                            for (int kw = 0; kw < 2; kw++) {
                                int inH = h * 2 + kh;
                                int inW = w * 2 + kw;
                                float val = input(n, c, inH, inW);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxIdx = input.index(n, c, inH, inW);
                                }
                            }
                        }
                        output(n, c, h, w) = maxVal;
                        maxIndices_[output.index(n, c, h, w)] = maxIdx;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(Tensor &output) {
        Tensor input = output;
        for (size_t i = 0; i < output.elements(); i++) {
            if (maxIndices_[i] != -1) {
                input.data[maxIndices_[i]] = output.data[i];
            }
        }
        return input;
    }
};

class Upsample2x2 {
private: 
    Shape inputShape_;
public: 
    Tensor forward(Tensor &input) {
        inputShape_ = { input.n, input.c, input.h, input.w };
        Tensor output(input.n, input.c, input.h * 2, input.w * 2);
        for(int n = 0; n < input.n; n++) {  
            for(int c = 0; c < input.c; c++) {
                for(int h = 0; h < input.h; h++) {
                    for(int w = 0; w < input.w; w++) {
                        float val = input(n, c, h, w);
                        for(int kh = 0; kh < 2; kh++) {
                            for(int kw = 0; kw < 2; kw++) {
                                output(n, c, h * 2 + kh, w * 2 + kw) = val;
                            }
                        }
                    }
                }
            }   
        }   
        return output;
    }

    Tensor backward(Tensor &output) {
        Tensor input(inputShape_.n, inputShape_.c, inputShape_.h, inputShape_.w);
        input.fill(0.0f);
        for(int n = 0; n < output.n; n++) {  
            for(int c = 0; c < output.c; c++) {
                for(int h = 0; h < output.h; h++) {
                    for(int w = 0; w < output.w; w++) {
                        input(n, c, h / 2, w / 2) += output(n, c, h, w);
                    }
                }
            }
        }
        return input;
    }
};  

class Conv2D {
private: 
    int inChannels_; 
    int outChannels_; 
    int kernelSize_; 
    int padding_; 
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> gradWeights_;
    std::vector<float> gradBias_;
    Tensor lastInput_;

    size_t weightIndex(int oc, int ic, int ky, int kx) {
        const size_t strideC = static_cast<size_t>(kernelSize_) * kernelSize_;
        const size_t strideIC = static_cast<size_t>(inChannels_) * strideC;
        return static_cast<size_t>(oc) * strideIC + static_cast<size_t>(ic) * strideC + static_cast<size_t>(ky) * kernelSize_ + kx;
    }

public: 
    Conv2D(int inChannels, int outChannels, int kernelSize, int padding) : inChannels_(inChannels), outChannels_(outChannels), kernelSize_(kernelSize), padding_(padding) {
        size_t count = static_cast<size_t>(outChannels_) * inChannels_ * kernelSize_ * kernelSize_;
        weights_.resize(count);
        bias_.assign(outChannels_, 0.0f);
        gradWeights_.resize(count);
        gradBias_.assign(outChannels_, 0.0f);
        lastInput_.resize(inChannels_, inChannels_, kernelSize_, kernelSize_);
        std::mt19937 rng(inChannels_ * 20 + outChannels_ * 25);\

        // https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        const float limit = std::sqrt(6.0f / (inChannels_ * kernelSize_ * kernelSize_ + outChannels_ * kernelSize_ * kernelSize_));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto &w : weights_)
            w = dist(rng);
    }

    Tensor forward(Tensor &input) {
        lastInput_ = input;
        Tensor output(input.n, outChannels_, input.h, input.w);

        for(int n = 0; n < input.n; n++) {  
            for(int oc = 0; oc < outChannels_; oc++) {
                for(int h = 0; h < output.h; h++) {
                    for(int w = 0; w < output.w; w++) {
                        float sum = bias_[oc];
                        for(int ic = 0; ic < inChannels_; ic++) {
                            for(int kh = 0; kh < kernelSize_; kh++) {
                                for(int kw = 0; kw < kernelSize_; kw++) {
                                    int inY = h + kh - padding_; 
                                    int inX = w + kw - padding_;
                                    if(inY < 0 || inY >= input.h || inX < 0 || inX >= input.w)
                                        continue;
                                    float inVal = input(n, ic, inY, inX);
                                    float wVal = weights_[weightIndex(oc, ic, kh, kw)];
                                    sum += inVal * wVal;
                                }
                            }
                        }
                        output(n, oc, h, w) = sum;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(Tensor &output, float learningRate) {
        Tensor input(lastInput_.n, lastInput_.c, lastInput_.h, lastInput_.w);
        input.fill(0.0f);
        std::fill(gradWeights_.begin(), gradWeights_.end(), 0.0f);
        std::fill(gradBias_.begin(), gradBias_.end(), 0.0f);
        
        for(int n = 0; n < output.n; n++) {  
            for(int oc = 0; oc < output.c; oc++) {
                for(int h = 0; h < output.h; h++) {
                    for(int w = 0; w < output.w; w++) {
                        float grad = output(n, oc, h, w);
                        gradBias_[oc] += grad;
                        for(int ic = 0; ic < inChannels_; ic++) {
                            for(int kh = 0; kh < kernelSize_; kh++) {
                                for(int kw = 0; kw < kernelSize_; kw++) {
                                    int inY = h + kh - padding_;
                                    int inX = w + kw - padding_;
                                    if(inY < 0 || inY >= input.h || inX < 0 || inX >= input.w)
                                        continue;
                                    float inVal = lastInput_(n, ic, inY, inX);
                                    size_t wIdx = weightIndex(oc, ic, kh, kw);
                                    gradWeights_[wIdx] += inVal * grad;
                                    input(n, ic, inY, inX) += weights_[wIdx] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        const float scale = 1.0f / static_cast<float>(output.n);
        for (size_t i = 0; i < weights_.size(); i++) {
            weights_[i] -= learningRate * gradWeights_[i] * scale;
        }
        for (int oc = 0; oc < outChannels_; oc++) {
            bias_[oc] -= learningRate * gradBias_[oc] * scale;
        }
        return input;
    }
};


class MSE {
private: 
    Tensor grad; 
public: 
    float forward(Tensor &input, Tensor &target) {
        if (input.elements() != target.elements())
            throw std::runtime_error("MSE: input and target size mismatch");

        grad.resize(input.n, input.c, input.h, input.w);
        float loss = 0.0f;
        size_t count = input.elements();
        for (size_t i = 0; i < count; i++) {
            float diff = input.data[i] - target.data[i];
            loss += diff * diff;
            grad.data[i] = 2.0f * diff / static_cast<float>(count);
        }
        return loss / static_cast<float>(count);
    }

    Tensor backward() {
        return grad;
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