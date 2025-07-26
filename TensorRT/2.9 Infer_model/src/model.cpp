#include <memory>
#include <iostream>
#include <string>
#include <type_traits>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include "argsParser.h"
#include "NvInferRuntime.h"

#include "../inc/model.hpp"
#include "../inc/utils.hpp"


class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char *msg) noexcept override{
        std::string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]:"  CLEAR;
            case Severity::kERROR:          str = RED    "[error]:"  CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]: "  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]: "  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]: "  CLEAR;

        }
        if (severity <= Severity::kINFO)
            std::cout << str << std::string(msg) << std::endl;
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using unique_ptr_with_deleter = std::unique_ptr<T, InferDeleter>;

Model::Model(std::string onnxPath){
    if (!fileExists(onnxPath)){
        LOGE("%s not found. Program terminated.", onnxPath.c_str());
        exit(1);
    }
    mOnnxPath = onnxPath;
    mEnginePath = getEnginePath(onnxPath);
}

bool Model::build(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = unique_ptr_with_deleter<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = unique_ptr_with_deleter<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = unique_ptr_with_deleter<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = unique_ptr_with_deleter<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    // 设置工作区内存池的最大内存为 256 MB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28);
    
    // 用parser解析onnx文件
    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    // 序列化engine
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    //
    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);
    // 通过IRuntime的反序列化api生成反序列化引擎
    mEngine            = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    
    // 通过 network-> 就可以看自动补全 有那些可以打印 打印的API开头是get
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();

    // 把优化前和优化后的各个层的信息打印出来
    LOG("Before TensorRT optimization");
    print_network(*network, false);
    LOG("");
    LOG("After TensorRT optimization");
    print_network(*network, true);

    mEngine.reset();
    return true;
};

bool Model::infer(){
    /*  
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输 
    */
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

     /* 反序列化从文件中读取的数据以unsigned char的vector保存*/
    std::vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);

    Logger logger;
    auto runtime     = unique_ptr_with_deleter<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = unique_ptr_with_deleter<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = unique_ptr_with_deleter<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 获取输入和输出张量的名称
    for (int i = 0; i < engine->getNbIOTensors(); i++){
        const char* input_name = engine->getIOTensorName(i);
        // bool isInput = engine->bindingIsInput(i);
        // 获取张量的形状
        auto input_dims = context->getTensorShape(input_name);

        LOG("input dim shape is:  %s", input_name, printDims(input_dims).c_str());
    }


    /* 2. host->device的数据传递 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* host memory上的数据*/
    float input_host[] = {0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552};
    float output_host[5];

    /* device memory上的数据*/
    float* input_device = nullptr;
    float* weight_device = nullptr;
    float* output_device = nullptr;

    int input_size = 10;
    int output_size = 5;

    /* 分配空间, 并传送数据从host到device*/
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&output_device, sizeof(output_host));
    cudaMemcpyAsync(input_device, input_host, sizeof(input_host), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[] = {input_device, output_device};
    bool success = context->executeV2((void**)bindings);
    if (!success) {
        LOGE("Failed to execute inference.");
        return false;
    }

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(output_host, output_device, sizeof(output_host), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    LOG("input data is:  %s", printTensor(input_host, input_size).c_str());
    LOG("output data is: %s", printTensor(output_host, output_size).c_str());
    LOG("finished inference");
    return true;
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {
    // ITensor, ILayer, INetwork
    // ICudaEngine, IExecutionContext, IBuilder

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    std::string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOG("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOG("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? mEngine->getNbLayers() : network.getNbLayers();
    LOG("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOG("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = unique_ptr_with_deleter<nvinfer1::IEngineInspector>(mEngine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            std::string info = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kONELINE);

            info = info.substr(0, info.size() - 1);
            LOG("layer_info: %s", info.c_str());
        }
    }
}

