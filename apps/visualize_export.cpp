#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

#include "core/Tensor.h"
#include "models/Autoencoder.h"
#include "dataloader/CifarDataLoader.h"
#include "utils/Serializer.h"

// Hàm tiện ích để lưu Tensor ra file nhị phân thô (Raw float) cho Python đọc
void save_tensor_raw(const Tensor& t, std::ofstream& out) {
    // Đảm bảo Tensor đang ở CPU
    Tensor cpu_t = t.to(DeviceType::CPU);
    out.write((char*)cpu_t.data_ptr(), cpu_t.numel() * sizeof(float));
}

int main(int argc, char** argv) {
    // 1. Cấu hình
    std::string data_path = "../data/cifar-10-batches-bin";
    std::string model_path = "weights/gpu_epoch_10.bin"; // File weight bạn muốn load
    int num_vis_samples = 20; // Số lượng ảnh muốn visualize

    if (argc > 1) model_path = argv[1];

    std::cout << ">> Loading model from: " << model_path << std::endl;

    // 2. Load Model
    Autoencoder model; // Mặc định khởi tạo trên CPU
    try {
        Serializer::load_model(model, model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }
    std::cout << ">> Model loaded successfully.\n";

    // 3. Load Data (Test Set)
    // Batch size = số lượng ảnh muốn hiện
    CifarDataLoader loader(data_path, CifarDataLoader::Split::Test, num_vis_samples);
    Batch batch = loader.nextBatch();
    Tensor inputs = batch.images; // [N, 3, 32, 32]

    // 4. Inference (Chạy trên CPU cho đơn giản vì chỉ cần vài ảnh)
    std::cout << ">> Running inference on CPU...\n";
    Tensor outputs = model.forward(inputs);

    // 5. Lưu kết quả ra file binary để Python đọc
    std::string out_file = "vis_data.bin";
    std::ofstream out(out_file, std::ios::binary);
    
    // Format file: [Num_Images] [Input_Raw_Floats] [Output_Raw_Floats]
    int N = inputs.sizes[0];
    out.write((char*)&N, sizeof(int)); // Ghi số lượng ảnh
    
    std::cout << ">> Exporting " << N << " images to " << out_file << "...\n";
    save_tensor_raw(inputs, out);
    save_tensor_raw(outputs, out);

    out.close();
    std::cout << ">> Done! Now run the Python script to visualize.\n";

    return 0;
}