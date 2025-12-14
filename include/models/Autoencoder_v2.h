#pragma once
#include <vector>
#include "../layers/ILayer.h"
#include "../layers/Conv2D_relu.h" // Sử dụng lớp Fused
#include "../layers/Conv2D.h"      // Sử dụng cho layer cuối (ko ReLU)
#include "../layers/MaxPool2D.h"
#include "../layers/Upsample.h"
#include "../layers/Sigmoid.h"

class Autoencoder_v2 {
public:
    // --- ENCODER ---
    Conv2D_relu enc_fused1; // Thay thế Conv + ReLU rời
    MaxPool2D   enc_pool1;

    Conv2D_relu enc_fused2;
    MaxPool2D   enc_pool2;

    // --- DECODER ---
    Conv2D_relu dec_fused1;
    Upsample    dec_up1;

    Conv2D_relu dec_fused2;
    Upsample    dec_up2;

    // Lớp cuối cùng thường không có ReLU mà dùng Sigmoid hoặc Linear
    Conv2D      dec_final; 
    Sigmoid     dec_sigmoid;

    Autoencoder_v2() 
        // Encoder Block 1
        : enc_fused1(3, 256, 3, 1, 1), 
          enc_pool1(2, 2),
        
        // Encoder Block 2
          enc_fused2(256, 128, 3, 1, 1),
          enc_pool2(2, 2),

        // Decoder Block 1
          dec_fused1(128, 128, 3, 1, 1),
          dec_up1(2),

        // Decoder Block 2
          dec_fused2(128, 256, 3, 1, 1),
          dec_up2(2),

        // Output Block (Raw Conv -> Sigmoid)
          dec_final(256, 3, 3, 1, 1),
          dec_sigmoid()
    {}

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        auto add_p = [&](Layer& l) {
            auto p = l.parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        
        add_p(enc_fused1);
        add_p(enc_fused2);
        add_p(dec_fused1);
        add_p(dec_fused2);
        add_p(dec_final); // Conv2D thường
        return params;
    }

    Tensor forward(const Tensor& x) {
        // Encoder
        Tensor out = enc_fused1.forward(x); // Đã bao gồm Conv+ReLU
        out = enc_pool1.forward(out);

        out = enc_fused2.forward(out);
        out = enc_pool2.forward(out);

        // Decoder
        out = dec_fused1.forward(out);
        out = dec_up1.forward(out);

        out = dec_fused2.forward(out);
        out = dec_up2.forward(out);

        out = dec_final.forward(out);
        out = dec_sigmoid.forward(out);
        
        return out;
    }

    void backward(const Tensor& grad_output) {
        // Decoder Backward
        Tensor d = dec_sigmoid.backward(grad_output);
        d = dec_final.backward(d);

        d = dec_up2.backward(d);
        d = dec_fused2.backward(d); 

        d = dec_up1.backward(d);
        d = dec_fused1.backward(d);

        // Encoder Backward
        d = enc_pool2.backward(d);
        d = enc_fused2.backward(d);

        d = enc_pool1.backward(d);
        d = enc_fused1.backward(d);
    }

    void to(DeviceType device) {
        enc_fused1.to(device);
        enc_fused2.to(device);
        dec_fused1.to(device);
        dec_fused2.to(device);
        dec_final.to(device);
    }
};