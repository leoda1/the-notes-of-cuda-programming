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
    



}