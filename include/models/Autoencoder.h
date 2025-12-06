#pragma once
#include <vector>
#include "layers/ILayer.h"
#include "layers/Conv2D.h"
#include "layers/ReLU.h"
#include "layers/MaxPool2D.h"
#include "layers/Upsample.h"
#include "layers/Sigmoid.h"
class Autoencoder {
public:
    // --- ENCODER LAYERS ---
    Conv2D enc_conv1;
    ReLU enc_relu1;
    MaxPool2D enc_pool1;
    
    Conv2D enc_conv2;
    ReLU enc_relu2;
    MaxPool2D enc_pool2;

    // --- DECODER LAYERS ---
    Conv2D dec_conv1;
    ReLU dec_relu1;
    Upsample dec_up1;

    Conv2D dec_conv2;
    ReLU dec_relu2;
    Upsample dec_up2;

    Conv2D dec_conv3; 
    Sigmoid dec_sigmoid;

    Autoencoder() 
        // Init Encoder
        : enc_conv1(3, 256, 3, 1, 1), 
          enc_pool1(2, 2),
          enc_conv2(256, 128, 3, 1, 1),
          enc_pool2(2, 2),
        
        // Init Decoder
          dec_conv1(128, 128, 3, 1, 1),
          dec_up1(2),
          dec_conv2(128, 256, 3, 1, 1),
          dec_up2(2),
          dec_conv3(256, 3, 3, 1, 1),
          dec_sigmoid()
    {}

    // Thu thập tất cả tham số của mạng để đưa cho Optimizer
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        auto add_p = [&](Layer& l) {
            auto p = l.parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        add_p(enc_conv1); add_p(enc_conv2);
        add_p(dec_conv1); add_p(dec_conv2); add_p(dec_conv3);
        return params;
    }

    Tensor forward(const Tensor& x) {
        // Encoder
        Tensor out = enc_conv1.forward(x);
        out = enc_relu1.forward(out);
        out = enc_pool1.forward(out); // -> (16, 16, 256)
        
        out = enc_conv2.forward(out);
        out = enc_relu2.forward(out);

        // latent space 
        out = enc_pool2.forward(out); // -> (8, 8, 128) 

        // Decoder
        out = dec_conv1.forward(out);
        out = dec_relu1.forward(out);
        out = dec_up1.forward(out);   // -> (16, 16, 128)

        out = dec_conv2.forward(out);
        out = dec_relu2.forward(out);
        out = dec_up2.forward(out);   // -> (32, 32, 256)

        out = dec_conv3.forward(out); // -> (32, 32, 3)
        out = dec_sigmoid.forward(out);
        return out;
    }

    void backward(const Tensor& grad_output) {
        // Decoder Backward
        Tensor d = dec_sigmoid.backward(grad_output);
        d = dec_conv3.backward(d);

        d = dec_up2.backward(d);
        d = dec_relu2.backward(d);
        d = dec_conv2.backward(d);

        d = dec_up1.backward(d);
        d = dec_relu1.backward(d);
        d = dec_conv1.backward(d);

        // Encoder Backward
        d = enc_pool2.backward(d);
        d = enc_relu2.backward(d);
        d = enc_conv2.backward(d);

        d = enc_pool1.backward(d);
        d = enc_relu1.backward(d);
        d = enc_conv1.backward(d);
    }
    
    // chuyển tham số sang GPU . có thể thay thế sau này bằng GPU <-> CPU
    void to(DeviceType device) {
        enc_conv1.to(device);
        enc_conv2.to(device);
        dec_conv1.to(device);
        dec_conv2.to(device);
        dec_conv3.to(device);
    }
};