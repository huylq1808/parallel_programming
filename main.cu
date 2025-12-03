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

__global__ void conv2dForwardKernel(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ bias, float *__restrict__ output, int N, int inC, int outC, int H, int W, int kSize, int padding)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n_oc = blockIdx.z;

    int n = n_oc / outC;
    int oc = n_oc % outC;

    if (x >= W || y >= H || n >= N)
        return;

    float sum = bias[oc];

    for (int ic = 0; ic < inC; ic++) {
        for (int kh = 0; kh < kSize; kh++) {
            for (int kw = 0; kw < kSize; kw++) {
                int inY = y + kh - padding;
                int inX = x + kw - padding;

                if (inY >= 0 && inY < H && inX >= 0 && inX < W) {
                    int inIdx = ((n * inC + ic) * H + inY) * W + inX;
                    int wIdx = ((oc * inC + ic) * kSize + kh) * kSize + kw;
                    sum += input[inIdx] * weights[wIdx];
                }
            }
        }
    }

    int outIdx = ((n * outC + oc) * H + y) * W + x;
    output[outIdx] = sum;
}

__global__ void conv2dBackwardInputKernel(const float *__restrict__ gradOutput, const float *__restrict__ weights, float *__restrict__ gradInput, int N, int inC, int outC, int H, int W, int kSize, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n_oc = blockIdx.z;

    int n = n_oc / inC;
    int ic = n_oc % inC;

    if (x >= W || y >= H || n >= N)
        return;

    float sum = 0.0f;

    for (int oc = 0; oc < outC; oc++) {
        for (int kh = 0; kh < kSize; kh++) {
            for (int kw = 0; kw < kSize; kw++) {
                int outY = y - kh + padding;
                int outX = x - kw + padding;

                if (outY >= 0 && outY < H && outX >= 0 && outX < W) {
                    int gradOutIdx = ((n * outC + oc) * H + outY) * W + outX;
                    int wIdx = ((oc * inC + ic) * kSize + kh) * kSize + kw;
                    sum += gradOutput[gradOutIdx] * weights[wIdx];
                }
            }
        }
    }

    int inIdx = ((n * inC + ic) * H + y) * W + x;
    gradInput[inIdx] = sum;
}

__global__ void conv2dBackwardWeightsKernel(const float *__restrict__ input, const float *__restrict__ gradOutput, float *__restrict__ gradWeights, int N, int inC, int outC, int H, int W, int kSize, int padding) {
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int k_idx = blockIdx.z;

    if (oc >= outC || ic >= inC || k_idx >= kSize * kSize)
        return;

    int kh = k_idx / kSize;
    int kw = k_idx % kSize;

    float gradW = 0.0f;

    for (int n = 0; n < N; n++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++){
                int inY = y + kh - padding;
                int inX = x + kw - padding;

                if (inY >= 0 && inY < H && inX >= 0 && inX < W) {
                    int inIdx = ((n * inC + ic) * H + inY) * W + inX;
                    int gradOutIdx = ((n * outC + oc) * H + y) * W + x;
                    gradW += input[inIdx] * gradOutput[gradOutIdx];
                }
            }
        }
    }

    int wIdx = ((oc * inC + ic) * kSize + kh) * kSize + kw;
    atomicAdd(&gradWeights[wIdx], gradW);
}

__global__ void conv2dBackwardBiasKernel(const float *__restrict__ gradOutput, float *__restrict__ gradBias, int N, int outC, int H, int W) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outC)
        return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int y = 0; y < H; y++) { 
            for (int x = 0; x < W; x++){
                int idx = ((n * outC + oc) * H + y) * W + x;
                sum += gradOutput[idx];
            }
        }
    }

    gradBias[oc] = sum;
}

__global__ void sgdUpdateKernel(float *__restrict__ weights, const float *__restrict__ gradients, float learningRate, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    weights[idx] -= learningRate * gradients[idx] * scale;
}

__global__ void zeroMemoryKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    data[idx] = 0.0f;
}

__global__ void sumReductionKernel(const float *__restrict__ input, float *__restrict__ output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, sdata[0]);
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

void testConv2DForward() {
    std::cout << "=== Testing Conv2D Forward ===\n";
    
    // Simple: N=1, inC=1, outC=1, H=4, W=4, kernel=3x3, padding=1
    const int N = 1, inC = 1, outC = 1, H = 4, W = 4, kSize = 3, padding = 1;
    const int inSize = N * inC * H * W;
    const int outSize = N * outC * H * W;
    const int wSize = outC * inC * kSize * kSize;
    
    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<float> h_weights(wSize, 1.0f / 9.0f);  // Average filter
    std::vector<float> h_bias = {0.0f};
    std::vector<float> h_output(outSize);
    
    float *d_input, *d_weights, *d_bias, *d_output;
    CHECK(cudaMalloc(&d_input, inSize * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, wSize * sizeof(float)));
    CHECK(cudaMalloc(&d_bias, outC * sizeof(float)));
    CHECK(cudaMalloc(&d_output, outSize * sizeof(float)));
    
    CHECK(cudaMemcpy(d_input, h_input.data(), inSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, h_weights.data(), wSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias.data(), outC * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, N * outC);
    conv2dForwardKernel<<<grid, block>>>(d_input, d_weights, d_bias, d_output, N, inC, outC, H, W, kSize, padding);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_output.data(), d_output, outSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Input (4x4):\n";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            printf("%.1f\t", h_input[i * W + j]);
        std::cout << "\n";
    }
    
    std::cout << "Output (3x3 avg filter, padding=1):\n";
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            printf("%.2f\t", h_output[i * W + j]);
        std::cout << "\n";
    }
    std::cout << "\n";
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

void testUpsample2x2() {
    std::cout << "=== Testing Upsample2x2 ===\n";
    
    const int N = 1, C = 1, inH = 2, inW = 2;
    const int outH = inH * 2, outW = inW * 2;
    const int inSize = N * C * inH * inW;
    const int outSize = N * C * outH * outW;
    
    std::vector<float> h_input = {1, 2, 3, 4};
    std::vector<float> h_output(outSize);
    
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, inSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output, outSize * sizeof(float)));
    
    CHECK(cudaMemcpy(d_input, h_input.data(), inSize * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, N * C);
    Upsample2x2Forward<<<grid, block>>>(d_input, d_output, N, C, inH, inW);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_output.data(), d_output, outSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Input (2x2):\n";
    for (int i = 0; i < inH; i++) {
        for (int j = 0; j < inW; j++)
            std::cout << h_input[i * inW + j] << "\t";
        std::cout << "\n";
    }
    
    std::cout << "Output (4x4 upsampled):\n";
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++)
            std::cout << h_output[i * outW + j] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void testSumReduction() {
    std::cout << "=== Testing Sum Reduction ===\n";
    
    const int size = 1024;
    std::vector<float> h_input(size);
    for (int i = 0; i < size; i++) h_input[i] = 1.0f;  // Sum should be 1024
    
    float h_output = 0.0f;
    
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, sizeof(float)));
    CHECK(cudaMemset(d_output, 0, sizeof(float)));
    
    CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sumReductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Sum of " << size << " ones = " << h_output << " (expected: " << size << ")\n\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void testMSELoss() {
    std::cout << "=== Testing MSE Loss ===\n";
    
    const int size = 4;
    std::vector<float> h_pred = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_target = {1.5f, 2.5f, 3.5f, 4.5f};  // All off by 0.5
    std::vector<float> h_sqErrors(size);
    std::vector<float> h_grad(size);
    
    float *d_pred, *d_target, *d_sqErrors, *d_grad;
    CHECK(cudaMalloc(&d_pred, size * sizeof(float)));
    CHECK(cudaMalloc(&d_target, size * sizeof(float)));
    CHECK(cudaMalloc(&d_sqErrors, size * sizeof(float)));
    CHECK(cudaMalloc(&d_grad, size * sizeof(float)));
    
    CHECK(cudaMemcpy(d_pred, h_pred.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_target, h_target.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    MSELossForward<<<gridSize, blockSize>>>(d_pred, d_target, d_sqErrors, d_grad, size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_sqErrors.data(), d_sqErrors, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_grad.data(), d_grad, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float mse = 0;
    for (auto e : h_sqErrors) mse += e;
    mse /= size;
    
    std::cout << "Pred:   "; for (auto v : h_pred) std::cout << v << " "; std::cout << "\n";
    std::cout << "Target: "; for (auto v : h_target) std::cout << v << " "; std::cout << "\n";
    std::cout << "MSE = " << mse << " (expected: 0.25)\n";
    std::cout << "Grad: "; for (auto v : h_grad) std::cout << v << " "; std::cout << "\n\n";
    
    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_sqErrors);
    cudaFree(d_grad);
}

int main(int argc, char **argv) {
    std::cout << "CUDA Kernel Tests\n";
    std::cout << "=================\n\n";
    
    printDeviceInfo();
    
    testReLU();
    testSigmoid();
    testMaxPool2x2();
    testUpsample2x2();
    testConv2DForward();
    testMSELoss();
    testSumReduction();
    
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