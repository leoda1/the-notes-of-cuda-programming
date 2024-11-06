#include <iostream>
#include <filesystem>
// file
#include <sstream>
#include <fstream>
#include <string>
#include "NvInfer.h"
//include
#include "../inc/utils.hpp"

bool fileExists(const std::string& filename){
    return std::filesystem::exists(filename);
}

bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size ){
    std::stringstream trtModelStream;
    std::ifstream cache(path);

    //将engine的内容写入trtModelStream中
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    // 计算model的大小
    trtModelStream.seekg(0, std::ios::end);
    size = trtModelStream.tellg();

    trtModelStream.seekg(0, std::ios::beg);

    // 将trtModelStream中的stream通过read函数写入modelMem中
    trtModelStream.read((char*)&data[0], size);
    return true;
}

std::string getEnginePath(std::string onnxPath){
    
    // 处理路径和文件名
    std::filesystem::path pathObj(onnxPath);
    // 获取父目录
    std::filesystem::path pathDir = pathObj.parent_path();
    // 构造新路径
    std::filesystem::path enginePath = pathDir / ("engine" + pathObj.stem().string() + ".engine");
    return enginePath.string();
}

std::vector<unsigned char> loadFile(const std::string& file){
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file: " << file << std::endl;
        return {};
    }

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<unsigned char> data(length);
    if (!in.read(reinterpret_cast<char*>(data.data()), length)) {
        std::cerr << "Failed to read file: " << file << std::endl;
        return {};
    }

    return data;
}

std::string printDims(const nvinfer1::Dims dims){
    int n = 0;
    char buff[100];
    std::string result;

    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1) {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

std::string printTensor(float* tensor, int size){
    int n = 0;
    char buff[100];
    std::string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
        if (i != size - 1){
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

std::string printTensorShape(nvinfer1::ITensor* tensor){
    std::string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
        str += std::to_string(dims.d[j]);
        if (j != dims.nbDims - 1) {
            str += " x ";
        }
    }
    str += "]";
    return str;
}

std::string getPrecision(nvinfer1::DataType type) {
    switch(type) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        default:                          return "unknown";
    }
}
