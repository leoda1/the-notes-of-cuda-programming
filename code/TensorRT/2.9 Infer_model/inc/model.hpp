#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// libraries
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <string>
#include <memory>


class Model{
public:
    Model(std::string onnxPath);
    bool build();
    bool infer();

private:
    std::string mOnnxPath;
    std::string mEnginePath;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
};


#endif // __MODEL_HPP__