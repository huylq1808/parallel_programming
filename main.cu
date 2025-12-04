#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

constexpr int BLOCK_SIZE = 32;

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

void xavierUniformInit(float* data, size_t count, int fanIn, int fanOut, unsigned seed) {
    std::mt19937 rng(seed);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (size_t i = 0; i < count; i++)
        data[i] = dist(rng);
}

struct GpuTensor {
    float *data = nullptr;
    int n = 0, c = 0, h = 0, w = 0;

    GpuTensor() = default;

    void allocate(int n_, int c_, int h_, int w_) {
        free(); // release old memory 
        n = n_;
        c = c_;
        h = h_;
        w = w_;
        size_t bytes = elements() * sizeof(float);
        CHECK(cudaMalloc(&data, bytes));
        CHECK(cudaMemset(data, 0, bytes));
    }

    void free() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
        n = c = h = w = 0;
    }

    size_t elements() const { return static_cast<size_t>(n) * c * h * w; }
    size_t bytes() const { return elements() * sizeof(float); }

    void copyFromHost(const Tensor &host) {
        if (elements() != host.elements())
            allocate(host.n, host.c, host.h, host.w);

        CHECK(cudaMemcpy(data, host.data.data(), bytes(), cudaMemcpyHostToDevice));
    }

    void copyToHost(Tensor &host) const {
        host.resize(n, c, h, w);
        CHECK(cudaMemcpy(host.data.data(), data, bytes(), cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (data && elements() > 0)
            CHECK(cudaMemset(data, 0, bytes()));
    }

    ~GpuTensor() { free(); }

    // knông cho copy, tránh bị double free
    GpuTensor(const GpuTensor &) = delete;
    GpuTensor &operator=(const GpuTensor &) = delete;

    
    GpuTensor(GpuTensor &&other) noexcept: data(other.data), n(other.n), c(other.c), h(other.h), w(other.w) {
        other.data = nullptr;
        other.n = other.c = other.h = other.w = 0;
    }

    GpuTensor &operator=(GpuTensor &&other) noexcept {
        if (this != &other) {
            free();
            data = other.data;
            n = other.n;
            c = other.c;
            h = other.h;
            w = other.w;
            other.data = nullptr;
            other.n = other.c = other.h = other.w = 0;
        }
        return *this;
    }
};

class GpuConv2D {
private: 
    int inChannels_, outChannels_, kernelSize_, padding_;
    float *d_weights_ = nullptr;
    float *d_bias_ = nullptr;
    float *d_gradWeights_ = nullptr;
    float *d_gradBias_ = nullptr;

    // Cached for backward
    float *d_lastInput_ = nullptr;
    int lastN_ = 0, lastH_ = 0, lastW_ = 0;

public:
    GpuConv2D(int inChannels, int outChannels, int kernelSize, int padding = 1) : inChannels_(inChannels), outChannels_(outChannels), kernelSize_(kernelSize), padding_(padding) {
        size_t weightCount = static_cast<size_t>(outChannels_) * inChannels_ * kernelSize_ * kernelSize_;

        CHECK(cudaMalloc(&d_weights_, weightCount * sizeof(float)));
        CHECK(cudaMalloc(&d_bias_, outChannels_ * sizeof(float)));
        CHECK(cudaMalloc(&d_gradWeights_, weightCount * sizeof(float)));
        CHECK(cudaMalloc(&d_gradBias_, outChannels_ * sizeof(float)));

        std::vector<float> h_weights(weightCount);
        std::vector<float> h_bias(outChannels_, 0.0f);

        unsigned seed = inChannels * 13 + outChannels * 17;
        int fanIn = inChannels * kernelSize * kernelSize;
        int fanOut = outChannels * kernelSize * kernelSize;
        xavierUniformInit(h_weights.data(), weightCount, fanIn, fanOut, seed);

        CHECK(cudaMemcpy(d_weights_, h_weights.data(), weightCount * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_, h_bias.data(), outChannels_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~GpuConv2D() {
        if (d_weights_)
            cudaFree(d_weights_);
        if (d_bias_)
            cudaFree(d_bias_);
        if (d_gradWeights_)
            cudaFree(d_gradWeights_);
        if (d_gradBias_)
            cudaFree(d_gradBias_);
    }

    void forward(const GpuTensor &input, GpuTensor &output) {
        lastN_ = input.n;
        lastH_ = input.h;
        lastW_ = input.w;
        d_lastInput_ = input.data;

        if (output.n != input.n || output.c != outChannels_ ||output.h != input.h || output.w != input.w)
            output.allocate(input.n, outChannels_, input.h, input.w);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((input.w + block.x - 1) / block.x, (input.h + block.y - 1) / block.y, input.n * outChannels_);

        conv2dForwardKernel<<<grid, block>>>(input.data, d_weights_, d_bias_, output.data,input.n, inChannels_, outChannels_, input.h, input.w, kernelSize_, padding_);
    }

    void backward(const GpuTensor &gradOutput, GpuTensor &gradInput, float learningRate) {
        size_t weightCount = static_cast<size_t>(outChannels_) * inChannels_ * kernelSize_ * kernelSize_;
        CHECK(cudaMemset(d_gradWeights_, 0, weightCount * sizeof(float)));
        CHECK(cudaMemset(d_gradBias_, 0, outChannels_ * sizeof(float)));
        
        if (gradInput.n != lastN_ || gradInput.c != inChannels_ || gradInput.h != lastH_ || gradInput.w != lastW_)
            gradInput.allocate(lastN_, inChannels_, lastH_, lastW_);
        
        gradInput.zero();

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridInput(
            (lastW_ + block.x - 1) / block.x,
            (lastH_ + block.y - 1) / block.y,
            lastN_ * inChannels_);

        conv2dBackwardInputKernel<<<gridInput, block>>>(gradOutput.data, d_weights_, gradInput.data,lastN_, inChannels_, outChannels_, lastH_, lastW_, kernelSize_, padding_);

        dim3 gridWeights(outChannels_, inChannels_, kernelSize_ * kernelSize_);
        conv2dBackwardWeightsKernel<<<gridWeights, 1>>>(d_lastInput_, gradOutput.data, d_gradWeights_, lastN_, inChannels_, outChannels_, lastH_, lastW_, kernelSize_, padding_);

        int biasBlocks = (outChannels_ + 255) / 256;
        conv2dBackwardBiasKernel<<<biasBlocks, 256>>>(gradOutput.data, d_gradBias_, lastN_, outChannels_, lastH_, lastW_);

        float scale = 1.0f / static_cast<float>(lastN_);
        int updateBlocks = (weightCount + 255) / 256;
        sgdUpdateKernel<<<updateBlocks, 256>>>(d_weights_, d_gradWeights_, learningRate, scale, weightCount);

        int biasUpdateBlocks = (outChannels_ + 255) / 256;
        sgdUpdateKernel<<<biasUpdateBlocks, 256>>>(d_bias_, d_gradBias_, learningRate, scale, outChannels_);
    }

};

class GpuReLU {
private:
    uint8_t *d_mask_ = nullptr;
    int maskSize_ = 0;

public: 
    void forward(const GpuTensor &input, GpuTensor &output) {
        int size = input.elements();

        if (maskSize_ < size) {
            if (d_mask_)
                cudaFree(d_mask_);
            CHECK(cudaMalloc(&d_mask_, size * sizeof(uint8_t)));
            maskSize_ = size;
        }

        if (output.elements() != input.elements())
            output.allocate(input.n, input.c, input.h, input.w);

        int blocks = (size + 255) / 256;
        ReLUForward<<<blocks, 256>>>(input.data, output.data, d_mask_, size);
    }

    void backward(const GpuTensor &gradOutput, GpuTensor &gradInput) {
        int size = gradOutput.elements();

        if (gradInput.elements() != size)
            gradInput.allocate(gradOutput.n, gradOutput.c, gradOutput.h, gradOutput.w);

        int blocks = (size + 255) / 256;
        ReLUBackward<<<blocks, 256>>>(gradOutput.data, gradInput.data, d_mask_, size);
    }

    ~GpuReLU() {
        if (d_mask_)
            cudaFree(d_mask_);
    }
};

class GpuSigmoid {
private:
    const GpuTensor *cachedOutput_ = nullptr;

public: 
    GpuSigmoid() = default;

    ~GpuSigmoid() {
        if (cachedOutput_)
            cachedOutput_ = nullptr;
    }

    void forward(const GpuTensor &input, GpuTensor &output) {
        if (output.elements() != input.elements())
            output.allocate(input.n, input.c, input.h, input.w);

        cachedOutput_ = &output;

        int size = input.elements();
        int blocks = (size + 255) / 256;
        SigmoidForward<<<blocks, 256>>>(input.data, output.data, size);
    }

    void backward(const GpuTensor &gradOutput, GpuTensor &gradInput) {
        int size = gradOutput.elements();

        if (gradInput.elements() != size)
            gradInput.allocate(gradOutput.n, gradOutput.c, gradOutput.h, gradOutput.w);

        int blocks = (size + 255) / 256;
        SigmoidBackward<<<blocks, 256>>>(gradOutput.data, cachedOutput_->data, gradInput.data, size);
    }
};

class GpuMaxPool2x2 {
private:
    int *d_maxIndices_ = nullptr;
    int maxIndicesSize_ = 0;
    int inN_ = 0, inC_ = 0, inH_ = 0, inW_ = 0;

public:
   void forward(const GpuTensor &input, GpuTensor &output) {
        int outH = input.h / 2;
        int outW = input.w / 2;

        inN_ = input.n;
        inC_ = input.c;
        inH_ = input.h;
        inW_ = input.w;

        if (output.n != input.n || output.c != input.c || output.h != outH || output.w != outW)
            output.allocate(input.n, input.c, outH, outW);

        int outSize = output.elements();
        if (maxIndicesSize_ < outSize) {
            if (d_maxIndices_)
                cudaFree(d_maxIndices_);
            CHECK(cudaMalloc(&d_maxIndices_, outSize * sizeof(int)));
            maxIndicesSize_ = outSize;
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, input.n * input.c);

        MaxPool2x2Forward<<<grid, block>>>(input.data, output.data, d_maxIndices_, input.n, input.c, input.h, input.w);
    }

    void backward(const GpuTensor &gradOutput, GpuTensor &gradInput) {
        if (gradInput.n != inN_ || gradInput.c != inC_ || gradInput.h != inH_ || gradInput.w != inW_)
            gradInput.allocate(inN_, inC_, inH_, inW_);
        
        gradInput.zero();

        int outSize = gradOutput.elements();
        int inSize = gradInput.elements();
        int blocks = (outSize + 255) / 256;

        MaxPool2x2Backward<<<blocks, 256>>>(gradOutput.data, d_maxIndices_, gradInput.data, outSize, inSize);
    }

    ~GpuMaxPool2x2() {
        if (d_maxIndices_)
            cudaFree(d_maxIndices_);
    }

};

class GpuUpsample2x2 {
private:
    int inN_ = 0, inC_ = 0, inH_ = 0, inW_ = 0;

public:
    void forward(const GpuTensor &input, GpuTensor &output) {
        int outH = input.h * 2;
        int outW = input.w * 2;

        inN_ = input.n;
        inC_ = input.c;
        inH_ = input.h;
        inW_ = input.w;

        if (output.n != input.n || output.c != input.c || output.h != outH || output.w != outW)
            output.allocate(input.n, input.c, outH, outW);
    

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, input.n * input.c);

        Upsample2x2Forward<<<grid, block>>>(input.data, output.data, input.n, input.c, input.h, input.w);
    }

    void backward(const GpuTensor &gradOutput, GpuTensor &gradInput) {
        if (gradInput.n != inN_ || gradInput.c != inC_ || gradInput.h != inH_ || gradInput.w != inW_)
            gradInput.allocate(inN_, inC_, inH_, inW_);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((inW_ + block.x - 1) / block.x, (inH_ + block.y - 1) / block.y,inN_ * inC_);

        Upsample2x2Backward<<<grid, block>>>(gradOutput.data, gradInput.data, inN_, inC_, inH_, inW_);
    }

};

class GpuMSELoss {
private:
    float *d_squaredErrors_ = nullptr;
    float *d_sum_ = nullptr;
    int squaredErrorsSize_ = 0;

public:
    float forward(const GpuTensor &prediction, const GpuTensor &target, GpuTensor &gradOutput) {
        int size = prediction.elements();

        if (squaredErrorsSize_ < size) {
            if (d_squaredErrors_)
                cudaFree(d_squaredErrors_);
            if (d_sum_)
                cudaFree(d_sum_);
            CHECK(cudaMalloc(&d_squaredErrors_, size * sizeof(float)));
            CHECK(cudaMalloc(&d_sum_, sizeof(float)));
            squaredErrorsSize_ = size;
        }

        if (gradOutput.elements() != size)
            gradOutput.allocate(prediction.n, prediction.c, prediction.h, prediction.w);

        int blocks = (size + 255) / 256;
        MSELossForward<<<blocks, 256>>>(prediction.data, target.data, d_squaredErrors_, gradOutput.data, size);

        CHECK(cudaMemset(d_sum_, 0, sizeof(float)));
        sumReductionKernel<<<blocks, 256, 256 * sizeof(float)>>>(
            d_squaredErrors_, d_sum_, size);

        float sum = 0.0f;
        CHECK(cudaMemcpy(&sum, d_sum_, sizeof(float), cudaMemcpyDeviceToHost));

        return sum / static_cast<float>(size);
    }

    ~GpuMSELoss() {
        if (d_squaredErrors_)
            cudaFree(d_squaredErrors_);
        if (d_sum_)
            cudaFree(d_sum_);
    }

};

class GpuAutoencoder {
private:
    GpuConv2D conv1_, conv2_, conv3_, conv4_;
    GpuReLU relu1_, relu2_, relu3_;
    GpuMaxPool2x2 pool_;
    GpuUpsample2x2 upsample_;
    GpuSigmoid sigmoid_;

    GpuTensor act1_, act2_, act3_, act4_, act5_, act6_, act7_;
    GpuTensor pooled_, upsampled_;

    GpuTensor grad1_, grad2_, grad3_, grad4_, grad5_;
    GpuTensor grad6_, grad7_, grad8_, grad9_, grad10_;

public:
    GpuAutoencoder()
        : conv1_(3, 32, 3, 1),
          conv2_(32, 32, 3, 1),
          conv3_(32, 32, 3, 1),
          conv4_(32, 3, 3, 1) {}

    void forward(const GpuTensor &input, GpuTensor &output) {
        // Encoder
        conv1_.forward(input, act1_);
        relu1_.forward(act1_, act2_);
        conv2_.forward(act2_, act3_);
        relu2_.forward(act3_, act4_);
        pool_.forward(act4_, pooled_);

        // Decoder
        upsample_.forward(pooled_, upsampled_);
        conv3_.forward(upsampled_, act5_);
        relu3_.forward(act5_, act6_);
        conv4_.forward(act6_, act7_);
        sigmoid_.forward(act7_, output);
    }

    void backward(const GpuTensor &gradOutput, float learningRate) {
        sigmoid_.backward(gradOutput, grad1_);
        conv4_.backward(grad1_, grad2_, learningRate);
        relu3_.backward(grad2_, grad3_);
        conv3_.backward(grad3_, grad4_, learningRate);
        upsample_.backward(grad4_, grad5_);

        pool_.backward(grad5_, grad6_);
        relu2_.backward(grad6_, grad7_);
        conv2_.backward(grad7_, grad8_, learningRate);
        relu1_.backward(grad8_, grad9_);
        conv1_.backward(grad9_, grad10_, learningRate);
    }

    const GpuTensor &getLatent() const { return pooled_; }
};

struct TrainConfig {
    int epochs = 20;
    int batchSize = 32;
    float learningRate = 0.001f;
    int logEvery = 100; 
};

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

// int main(int argc, char **argv) {
//     std::cout << "CUDA Kernel Tests\n";
//     std::cout << "=================\n\n";
    
//     printDeviceInfo();
    
//     testReLU();
//     testSigmoid();
//     testMaxPool2x2();
//     testUpsample2x2();
//     testConv2DForward();
//     testMSELoss();
//     testSumReduction();
    
//     std::cout << "All tests completed\n";
//     return 0;
// }

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cifar_root> [epochs] [batch_size] [learning_rate]\n";
        return 1;
    }

    printDeviceInfo();

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

        GpuAutoencoder model;
        GpuMSELoss loss;
        GpuTensor d_input, d_output, d_gradOutput;
        double totalTrainTimeMs = 0.0;

        std::cout << "\n========== GPU TRAINING ==========\n";
        std::cout << "Hyperparameters:\n";
        std::cout << "Batch size:     " << config.batchSize << "\n";
        std::cout << "Epochs:         " << config.epochs << "\n";
        std::cout << "Learning rate:  " << config.learningRate << "\n";
        std::cout << "Train samples:  " << trainLoader.numSamples() << "\n";
        std::cout << "Test samples:   " << testLoader.numSamples() << "\n";
        std::cout << "=============================================\n\n";

        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            std::cout << "Epoch " << epoch << "/" << config.epochs << "\n";
            trainLoader.startEpoch(true);
            size_t steps = (trainLoader.numSamples() + config.batchSize - 1) / config.batchSize;
            double epochLoss = 0.0;
            size_t seenSamples = 0;

            GpuTimer epochTimer;
            epochTimer.Start();

            for (size_t step = 0; step < steps; step++) {
                Batch batch = trainLoader.nextBatch(config.batchSize);

                // Copy to GPU
                d_input.copyFromHost(batch.images);

                // Forward
                model.forward(d_input, d_output);

                // Copy target to GPU (same as input for autoencoder)
                GpuTensor d_target;
                d_target.copyFromHost(batch.images);

                // Compute loss
                float batchLoss = loss.forward(d_output, d_target, d_gradOutput);

                // Backward
                model.backward(d_gradOutput, config.learningRate);

                epochLoss += batchLoss * batch.images.n;
                seenSamples += batch.images.n;

                if ((step + 1) % config.logEvery == 0 || step == 0 || step + 1 == steps) {
                    std::cout << "    batch " << (step + 1) << "/" << steps
                              << " mse=" << std::fixed << std::setprecision(6) << batchLoss << "\n";
                }
            }

            epochTimer.Stop();
            float epochMs = epochTimer.Elapsed();
            totalTrainTimeMs += epochMs;

            float avgLoss = epochLoss / seenSamples;
            float throughput = seenSamples / (epochMs / 1000.0f);
            std::cout << std::fixed << std::setprecision(6) << "train_loss=" << avgLoss << "\n";
            std::cout << std::fixed << std::setprecision(2) << "epoch_time=" << epochMs << " ms " << "throughput=" << throughput << " img/s\n\n";
        }

        std::cout << "========== GPU SUMMARY ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total training time:    " << totalTrainTimeMs / 1000.0 << " seconds\n";
        std::cout << "Average epoch time:     " << totalTrainTimeMs / config.epochs << " ms\n";
        std::cout << "Average throughput:     " << (trainLoader.numSamples() * config.epochs) / (totalTrainTimeMs / 1000.0) << " img/s\n";
        std::cout << "==========================================\n";
    }

    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}