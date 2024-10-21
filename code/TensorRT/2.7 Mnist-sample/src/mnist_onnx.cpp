#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

// using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;
const std::string gSampleName = "TensorRT.sample_onnx_mnist_cn";

class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {}
    bool build();
    bool infer();
private:
    samplesCommon::OnnxSampleParams mParams;
    nvinfer1::Dims                  mInputDims;
    nvinfer1::Dims                  mOutputDims;
    int mNumber{0};

    /* use smartpointers to point the engine */
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    /* create grid */
    bool constructNetwork(
        SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    bool processInput(const samplesCommon::BufferManager& buffers);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

void printHelpInfo()
{

}

int main(int argc, char** argv)
{
    // get onnx -> parse onnx to generate engine -> inference -> print result
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }

    // create logger
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);

    //create sample object
    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build())
    {
        return 
    }




}