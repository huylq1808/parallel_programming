#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

struct CpuTimer {
    std::chrono::steady_clock::time_point start_;
    
    void start() { start_ = std::chrono::steady_clock::now(); }
    
    double elapsedMs() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
};

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
            for (int i = 1; i <= 1; i++) {
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
        Tensor input(inputShape_.n, inputShape_.c, inputShape_.h, inputShape_.w);
        input.fill(0.0f);
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

class Flatten {
private: 
    Shape inputShape_;
    Tensor output_; 
public: 
    Tensor forward(Tensor &input) {
        inputShape_ = { input.n, input.c, input.h, input.w };
        Tensor output(input.n, input.c * input.h * input.w, 1, 1);
        output.data = input.data; 
        output_ = output;
        return output;
    }

    Tensor backward(Tensor &output) {
        Tensor input(inputShape_.n, inputShape_.c, inputShape_.h, inputShape_.w);
        input.data = output.data;
        return input;
    }

    const Tensor &output() const {
        return output_;
    }
};

class Reshape {
private: 
    Shape target_;
    Shape lastShape_;
public: 
    Reshape(int channels, int height, int width) : target_({ 0, channels, height, width }) {}
    Tensor forward(Tensor &input) {
        lastShape_ = { input.n, input.c, input.h, input.w };
        Tensor output(input.n, target_.c, target_.h, target_.w);
        if(input.elements() != output.elements())
            throw std::runtime_error("Reshape: input and output size mismatch");
        output.data = input.data;
        return output;
    }   

    Tensor backward(Tensor &output) {
        Tensor input(lastShape_.n, lastShape_.c, lastShape_.h, lastShape_.w);
        if(input.elements() != output.elements())
            throw std::runtime_error("Reshape: input and output size mismatch");
        input.data = output.data;
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

    const Tensor &backward() const {
        return grad;
    }
};

class Sigmoid {
private:
    Tensor outputCache_;

public:
    Tensor forward(Tensor &input) {
        outputCache_ = input;
        for (auto &val : outputCache_.data) {
            val = 1.0f / (1.0f + std::exp(-val));
        }
        return outputCache_;
    }

    Tensor backward(const Tensor &output) {
        Tensor input = output;
        for (size_t i = 0; i < input.elements(); i++) {
            float sig = outputCache_.data[i];
            input.data[i] *= sig * (1.0f - sig);
        }
        return input;
    }
};

class Autoencoder {
private:
    Conv2D conv1_;
    Conv2D conv2_;
    Conv2D conv3_;
    Conv2D conv4_;
    ReLU relu1_;
    ReLU relu2_;
    ReLU relu3_;
    MaxPool2x2 pool_;
    Upsample2x2 upsample_;
    Flatten flatten_;
    Reshape reshape_;
    Sigmoid sigmoid_;
    Tensor latent_;
    Tensor lastReconstruction_;

    Tensor runEncoder(Tensor &input) {
        Tensor x = conv1_.forward(input); 
        x = relu1_.forward(x);
        x = conv2_.forward(x);
        x = relu2_.forward(x);
        x = pool_.forward(x);
        x = flatten_.forward(x);
        latent_ = x;
        return x;
    }

public:
    Autoencoder(): 
        conv1_(3, 32, 3, 1),
        conv2_(32, 32, 3, 1),
        conv3_(32, 32, 3, 1),
        conv4_(32, 3, 3, 1),
        reshape_(32, 16, 16),
        sigmoid_() {}

    Tensor forward(Tensor &input) {
        Tensor latent = runEncoder(input);
        Tensor x = reshape_.forward(latent);
        x = upsample_.forward(x);
        x = conv3_.forward(x);
        x = relu3_.forward(x);
        x = conv4_.forward(x);
        x = sigmoid_.forward(x);
        lastReconstruction_ = x;
        return lastReconstruction_;
    }

    void backward(const Tensor &output, float learningRate) {
        Tensor grad = sigmoid_.backward(output);
        grad = conv4_.backward(grad, learningRate);
        grad = relu3_.backward(grad);
        grad = conv3_.backward(grad, learningRate);
        grad = upsample_.backward(grad);
        grad = reshape_.backward(grad);
        grad = flatten_.backward(grad);
        grad = pool_.backward(grad);
        grad = relu2_.backward(grad);
        grad = conv2_.backward(grad, learningRate);
        grad = relu1_.backward(grad);
        (void)conv1_.backward(grad, learningRate);
    }

    Tensor encodeBatch(Tensor &input) {
        return runEncoder(input);
    }

    const Tensor &latentCache() const {
        return latent_;
    }

};

struct TrainConfig {
    int epochs = 20;
    int batchSize = 32;
    float learningRate = 1e-3f;
    int logEvery = 50;
};

struct EpochStats {
    float loss;
    double epochTimeMs;
    double imagesPerSec;
    size_t samplesProcessed;
};

EpochStats runEpoch(Autoencoder &model, CifarDataLoader &loader, int batchSize, float learningRate, int logEvery) {
    MSE loss;
    loader.startEpoch(true);
    const size_t steps = (loader.numSamples() + static_cast<size_t>(batchSize) - 1) / batchSize;
    double accumLoss = 0.0;
    size_t seenSamples = 0;
    const int safeLogEvery = std::max(1, logEvery);

    CpuTimer epochTimer;
    epochTimer.start();

    for (size_t step = 0; step < steps; step++) {
        Batch batch = loader.nextBatch(batchSize);
        Tensor recon = model.forward(batch.images);
        float batchLoss = loss.forward(recon, batch.images);
        model.backward(loss.backward(), learningRate);
        accumLoss += static_cast<double>(batchLoss) * batch.images.n;
        seenSamples += batch.images.n;

        if (((step + 1) % safeLogEvery) == 0 || step == 0 || step + 1 == steps) {
            std::cout << "    batch " << (step + 1) << "/" << steps
                      << " mse=" << std::fixed << std::setprecision(6) << batchLoss << "\n";
        }
    }

    double epochMs = epochTimer.elapsedMs();
    double imagesPerSec = seenSamples / (epochMs / 1000.0);

    return EpochStats{
        static_cast<float>(accumLoss / seenSamples),
        epochMs,
        imagesPerSec,
        seenSamples};
}

float evaluate(Autoencoder &model, CifarDataLoader &loader, int batchSize) {
    MSE loss;
    loader.startEpoch(false);
    const size_t steps = (loader.numSamples() + static_cast<size_t>(batchSize) - 1) / batchSize;
    double accumLoss = 0.0;
    size_t seenSamples = 0;
    for (size_t step = 0; step < steps; step++) {
        Batch batch = loader.nextBatch(batchSize);
        Tensor recon = model.forward(batch.images);
        float batchLoss = loss.forward(recon, batch.images);
        accumLoss += static_cast<double>(batchLoss) * batch.images.n;
        seenSamples += batch.images.n;
    }
    return static_cast<float>(accumLoss / static_cast<double>(seenSamples));
}

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

    TrainConfig config; 
    if (argc > 2) config.epochs = std::stoi(argv[2]);
    if (argc > 3) config.batchSize = std::stoi(argv[3]);
    if (argc > 4) config.learningRate = std::stof(argv[4]);

    try {
        std::cout << "Loading CIFAR-10 from " << cifarRoot << "\n";
        CifarDataLoader trainLoader(cifarRoot, CifarDataLoader::Split::Train);
        CifarDataLoader testLoader(cifarRoot, CifarDataLoader::Split::Test);
        std::cout << "Train samples: " << trainLoader.numSamples() << ", Test samples: " << testLoader.numSamples() << "\n";

        Autoencoder model; 
        double totalTrainTime = 0;
        std::vector<EpochStats> epochStats;
        
        std::cout << "\n========== CPU BASELINE TRAINING ==========\n";
        std::cout << "Hyperparameters:\n";
        std::cout << "Batch size:     " << config.batchSize << "\n";
        std::cout << "Epochs:         " << config.epochs << "\n";
        std::cout << "Learning rate:  " << config.learningRate << "\n";
        std::cout << "Train samples:  " << trainLoader.numSamples() << "\n";
        std::cout << "Test samples:   " << testLoader.numSamples() << "\n";
        std::cout << "=============================================\n\n";
        
        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            std::cout << "Epoch " << epoch << "/" << config.epochs << "\n";
            EpochStats stats = runEpoch(model, trainLoader, config.batchSize, config.learningRate, config.logEvery);
            float valLoss = evaluate(model, testLoader, config.batchSize);
            
            totalTrainTime += stats.epochTimeMs;
            epochStats.push_back(stats);
            
            std::cout << std::fixed << std::setprecision(6) << "train_loss=" << stats.loss << " " << "val_loss=" << valLoss << "\n";
            std::cout << std::fixed << std::setprecision(2) << "epoch_time=" << stats.epochTimeMs << " ms " << "throughput=" << stats.imagesPerSec << " img/s\n\n";
        }

        std::cout << "========== CPU BASELINE SUMMARY ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total training time:    " << totalTrainTime / 1000.0 << " seconds\n";
        std::cout << "Average epoch time:     " << totalTrainTime / config.epochs << " ms\n";
        std::cout << "Average throughput:     " << (trainLoader.numSamples() * config.epochs) / (totalTrainTime / 1000.0) << " img/s\n";
        std::cout << "Final train loss:       " << std::setprecision(6) << epochStats.back().loss << "\n";
        std::cout << "==========================================\n";
    }

    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}