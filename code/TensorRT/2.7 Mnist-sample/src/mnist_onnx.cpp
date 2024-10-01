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

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

int main(int argc, char** argv)
{
    // get onnx -> parse onnx to generate engine -> inference -> print result
    samplesCommon::Args args;

}