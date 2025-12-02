#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
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

// https://stackoverflow.com/questions/43235899/cuda-restrict-tag-usage
__global__ void ReLUForward(const float* __restrict__ input, float* __restrict__ output, uint8_t* __restrict__ mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return; 
    
    float val = input[idx]; 
    bool active = val > 0.0f; 
    output[idx] = active ? val : 0.0f;
    mask[idx] = active ? 1 : 0;
}

__global__ void ReLUBackward(const float* __restrict__ grad, float* __restrict__ output, uint8_t* __restrict__ mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return; 
    
    output[idx] = grad[idx] * static_cast<float>(mask[idx]);
}


__global__ void SigmoidForward(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

__global__ void SigmoidBackward(const float* __restrict__ gradOutput, const float* __restrict__ sigmoidOutput, float* __restrict__ gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float sig = sigmoidOutput[idx];
    gradInput[idx] = gradOutput[idx] * sig * (1.0f - sig);
}

__global__ void MaxPool2x2Forward(const float* __restrict__ input, float* __restrict__ output, int* __restrict__ maxIndices, int N, int C, int H, int W) {
    int outW = W / 2;
    int outH = H / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    int n = nc / C;
    int c = nc % C;

    if (x >= outW || y >= outH || n >= N)
        return;

    int baseY = y * 2;
    int baseX = x * 2;

    float maxVal = -1e30f;
    int maxIdx = -1;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int inY = baseY + dy;
            int inX = baseX + dx;
            int inIdx = ((n * C + c) * H + inY) * W + inX;
            float val = input[inIdx];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = inIdx;
            }
        }
    }

    int outIdx = ((n * C + c) * outH + y) * outW + x;
    output[outIdx] = maxVal;
    maxIndices[outIdx] = maxIdx;
}

__global__ void MaxPool2x2Backward(const float* __restrict__ gradOutput, const int* __restrict__ maxIndices, float* __restrict__ gradInput, int outSize, int inSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outSize)
        return;

    int maxIdx = maxIndices[idx];
    if (maxIdx >= 0 && maxIdx < inSize)
        atomicAdd(&gradInput[maxIdx], gradOutput[idx]);
}

__global__ void Upsample2x2Forward(const float* __restrict__ input, float* __restrict__ output, int N, int C, int inH, int inW) {
    int outH = inH * 2;
    int outW = inW * 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    int n = nc / C;
    int c = nc % C;

    if (x >= outW || y >= outH || n >= N)
        return;

    int srcY = y / 2;
    int srcX = x / 2;

    int inIdx = ((n * C + c) * inH + srcY) * inW + srcX;
    int outIdx = ((n * C + c) * outH + y) * outW + x;

    output[outIdx] = input[inIdx];
}

__global__ void Upsample2x2Backward(const float* __restrict__ gradOutput, float* __restrict__ gradInput, int N, int C, int inH, int inW) {
    int outH = inH * 2;
    int outW = inW * 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    int n = nc / C;
    int c = nc % C;

    if (x >= inW || y >= inH || n >= N)
        return;

    float sum = 0.0f;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int outY = y * 2 + dy;
            int outX = x * 2 + dx;
            int outIdx = ((n * C + c) * outH + outY) * outW + outX;
            sum += gradOutput[outIdx];
        }
    }

    int inIdx = ((n * C + c) * inH + y) * inW + x;
    gradInput[inIdx] = sum;
}

__global__ void MSELossForward(const float* __restrict__ prediction, const float* __restrict__ target, float* __restrict__ squaredErrors, float* __restrict__ gradOutput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float diff = prediction[idx] - target[idx];
    squaredErrors[idx] = diff * diff;
    gradOutput[idx] = (2.0f / static_cast<float>(size)) * diff;
}

// struct TrainConfig {
//     int epochs = 10;
//     int batchSize = 64;
//     float learningRate = 0.001f;
//     int logEvery = 100; 
// };

void printDeviceInfo() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "========== GPU INFO ==========\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Shared memory/block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "==============================\n\n";
}


void testReLU() {
    std::cout << "=== Testing ReLU ===\n";
    
    const int size = 8;
    std::vector<float> h_input = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, -0.5f, 3.0f};
    std::vector<float> h_output(size);
    std::vector<uint8_t> h_mask(size);
    
    // Allocate device memory
    float *d_input, *d_output;
    uint8_t *d_mask;
    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    CHECK(cudaMalloc(&d_mask, size * sizeof(uint8_t)));
    
    // Copy input to device
    CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    ReLUForward<<<gridSize, blockSize>>>(d_input, d_output, d_mask, size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_mask.data(), d_mask, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "Input:  ";
    for (auto v : h_input) std::cout << v << " ";
    std::cout << "\nOutput: ";
    for (auto v : h_output) std::cout << v << " ";
    std::cout << "\nMask:   ";
    for (auto v : h_mask) std::cout << (int)v << " ";
    std::cout << "\n\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

void testMaxPool2x2() {
    std::cout << "=== Testing MaxPool2x2 ===\n";
    
    // Simple 1x1x4x4 input (N=1, C=1, H=4, W=4)
    const int N = 1, C = 1, H = 4, W = 4;
    const int outH = H / 2, outW = W / 2;
    const int inSize = N * C * H * W;
    const int outSize = N * C * outH * outW;
    
    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<float> h_output(outSize);
    std::vector<int> h_indices(outSize);
    
    float *d_input, *d_output;
    int *d_indices;
    CHECK(cudaMalloc(&d_input, inSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output, outSize * sizeof(float)));
    CHECK(cudaMalloc(&d_indices, outSize * sizeof(int)));
    
    CHECK(cudaMemcpy(d_input, h_input.data(), inSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel (2D grid for spatial, Z for batch*channels)
    dim3 block(16, 16);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, N * C);
    MaxPool2x2Forward<<<grid, block>>>(d_input, d_output, d_indices, N, C, H, W);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_output.data(), d_output, outSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_indices.data(), d_indices, outSize * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Input (4x4):\n";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            std::cout << h_input[i * W + j] << "\t";
        std::cout << "\n";
    }
    
    std::cout << "Output (2x2 pooled):\n";
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++)
            std::cout << h_output[i * outW + j] << "\t";
        std::cout << "\n";
    }
    
    std::cout << "Max indices:\n";
    for (auto idx : h_indices) std::cout << idx << " ";
    std::cout << "\n\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
}

void testSigmoid() {
    std::cout << "=== Testing Sigmoid ===\n";
    
    const int size = 5;
    std::vector<float> h_input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> h_output(size);
    
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    SigmoidForward<<<gridSize, blockSize>>>(d_input, d_output, size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Input:  ";
    for (auto v : h_input) std::cout << v << " ";
    std::cout << "\nOutput: ";
    for (auto v : h_output) std::cout << v << " ";
    std::cout << "\n(Expected: ~0.12, ~0.27, 0.5, ~0.73, ~0.88)\n\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    std::cout << "CUDA Kernel Tests\n";
    std::cout << "=================\n\n";
    
    testReLU();
    testSigmoid();
    testMaxPool2x2();
    
    std::cout << "All tests completed\n";
    return 0;
}

// int main(int argc, char **argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <cifar_root> [epochs] [batch_size] [learning_rate]\n";
//         return 1;
//     }

//     const std::filesystem::path cifarRoot = argv[1];
//     if (!std::filesystem::exists(cifarRoot)) {
//         std::cerr << "Dataset folder not found: " << cifarRoot << "\n";
//         return 1;
//     }

//     TrainConfig config; 
//     if (argc > 2) config.epochs = std::stoi(argv[2]);
//     if (argc > 3) config.batchSize = std::stoi(argv[3]);
//     if (argc > 4) config.learningRate = std::stof(argv[4]);

//     try {
//         std::cout << "Loading CIFAR-10 from " << cifarRoot << "\n";
//         CifarDataLoader trainLoader(cifarRoot, CifarDataLoader::Split::Train);
//         CifarDataLoader testLoader(cifarRoot, CifarDataLoader::Split::Test);
//         std::cout << "Train samples: " << trainLoader.numSamples() << ", Test samples: " << testLoader.numSamples() << "\n";

//         Autoencoder model; 
//         double totalTrainTime = 0;
//         std::vector<EpochStats> epochStats;
        
//         std::cout << "\n========== CPU BASELINE TRAINING ==========\n";
//         std::cout << "Hyperparameters:\n";
//         std::cout << "Batch size:     " << config.batchSize << "\n";
//         std::cout << "Epochs:         " << config.epochs << "\n";
//         std::cout << "Learning rate:  " << config.learningRate << "\n";
//         std::cout << "Train samples:  " << trainLoader.numSamples() << "\n";
//         std::cout << "Test samples:   " << testLoader.numSamples() << "\n";
//         std::cout << "=============================================\n\n";
        
//         for (int epoch = 1; epoch <= config.epochs; epoch++) {
//             std::cout << "Epoch " << epoch << "/" << config.epochs << "\n";
//             EpochStats stats = runEpoch(model, trainLoader, config.batchSize, config.learningRate, config.logEvery);
//             float valLoss = evaluate(model, testLoader, config.batchSize);
            
//             totalTrainTime += stats.epochTimeMs;
//             epochStats.push_back(stats);
            
//             std::cout << std::fixed << std::setprecision(6) << "train_loss=" << stats.loss << " " << "val_loss=" << valLoss << "\n";
//             std::cout << std::fixed << std::setprecision(2) << "epoch_time=" << stats.epochTimeMs << " ms " << "throughput=" << stats.imagesPerSec << " img/s\n\n";
//         }

//         std::cout << "========== CPU BASELINE SUMMARY ==========\n";
//         std::cout << std::fixed << std::setprecision(2);
//         std::cout << "Total training time:    " << totalTrainTime / 1000.0 << " seconds\n";
//         std::cout << "Average epoch time:     " << totalTrainTime / config.epochs << " ms\n";
//         std::cout << "Average throughput:     " << (trainLoader.numSamples() * config.epochs) / (totalTrainTime / 1000.0) << " img/s\n";
//         std::cout << "Final train loss:       " << std::setprecision(6) << epochStats.back().loss << "\n";
//         std::cout << "==========================================\n";
//     }

//     catch (const std::exception &e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 1;
//     }

//     return 0;
// }