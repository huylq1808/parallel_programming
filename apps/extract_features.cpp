#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include "core/Tensor.h"
#include "core/CheckError.h"
#include "dataloader/CifarDataLoader.h"
#include "utils/Serializer.h"
#include "models/Autoencoder.h"
void save_features(const std::vector<float> &features, const std::vector<int> &labels, int num_samples, int feature_dim, const std::string &filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open: " + filepath);

    out.write((char *)&num_samples, sizeof(int));
    out.write((char *)&feature_dim, sizeof(int));

    out.write((char *)features.data(), features.size() * sizeof(float));

    out.write((char *)labels.data(), labels.size() * sizeof(int));

    std::cout << "Saved " << num_samples << " samples to " << filepath << std::endl;
}

void extract_features(Autoencoder &model, CifarDataLoader &loader, std::vector<float> &all_features, std::vector<int> &all_labels, int feature_dim)
{
    loader.startEpoch(false); 

    int batch_count = 0;
    while (loader.hasNext())
    {
        Batch batch = loader.nextBatch();

        Tensor images_gpu = batch.images.to(DeviceType::CUDA);

        Tensor features = model.encode(images_gpu);

        Tensor features_flat = features.view({features.sizes[0], feature_dim});

        Tensor features_cpu = features_flat.to(DeviceType::CPU);

        float *feat_ptr = (float *)features_cpu.data_ptr();
        float *label_ptr = (float *)batch.labels.data_ptr();
        int N = features_cpu.sizes[0];

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < feature_dim; ++j)
            {
                all_features.push_back(feat_ptr[i * feature_dim + j]);
            }
            all_labels.push_back(static_cast<int>(label_ptr[i])); 
        }

        batch_count++;
        if (batch_count % 50 == 0)
        {
            std::cout << "\rProcessed " << batch_count << " batches..." << std::flush;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    // Configuration
    std::string data_path = "../data/cifar-10-batches-bin";
    std::string weights_path = "weights/gpu_epoch_20.bin";
    int batch_size = 64;
    int feature_dim = 128 * 8 * 8; 

    if (argc >= 2)
        weights_path = argv[1];

    std::cout << "=== Feature Extraction ===" << std::endl;
    std::cout << "Weights: " << weights_path << std::endl;
    std::cout << "Feature dim: " << feature_dim << std::endl;

    // Check paths
    if (!std::filesystem::exists(data_path))
    {
        std::cerr << "Error: Data not found: " << data_path << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(weights_path))
    {
        std::cerr << "Error: Weights not found: " << weights_path << std::endl;
        return -1;
    }

    Autoencoder model;
    model.to(DeviceType::CUDA);
    Serializer::load_model(model, weights_path);
    std::cout << "Model loaded successfully!" << std::endl;

    std::cout << "\n--- Extracting Training Features ---" << std::endl;
    CifarDataLoader trainLoader(data_path, CifarDataLoader::Split::Train, batch_size, 50000);

    std::vector<float> train_features;
    std::vector<int> train_labels;
    train_features.reserve(50000 * feature_dim);
    train_labels.reserve(50000);

    extract_features(model, trainLoader, train_features, train_labels, feature_dim);
    save_features(train_features, train_labels, 50000, feature_dim, "train_features.bin");

    std::cout << "\n--- Extracting Test Features ---" << std::endl;
    CifarDataLoader testLoader(data_path, CifarDataLoader::Split::Test, batch_size, 10000);

    std::vector<float> test_features;
    std::vector<int> test_labels;
    test_features.reserve(10000 * feature_dim);
    test_labels.reserve(10000);

    extract_features(model, testLoader, test_features, test_labels, feature_dim);
    save_features(test_features, test_labels, 10000, feature_dim, "test_features.bin");

    std::cout << "Done\n";

    return 0;
}