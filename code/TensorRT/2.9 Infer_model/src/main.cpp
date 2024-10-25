#include <iostream>
#include <memory>

#include "../inc/model.hpp"
#include "../inc/utils.hpp"


int main(int argc, char const* argv[])
{
    Model model("models/onnx/infer.onnx");
    if (!model.build()){
        LOGE("Failed to build model");
        return -1;
    }
    if (!model.infer()){
        LOGE("Failed to infer model");
        return -1;
    }
}
