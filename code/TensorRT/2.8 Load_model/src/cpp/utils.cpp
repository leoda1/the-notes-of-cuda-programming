#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include "NvInfer.h"
#include "utils.hpp"

// 移除 using namespace std;

bool fileExists(const std::string& fileName) {
    return std::filesystem::exists(fileName);
}

bool fileRead(const std::string& path, std::vector<unsigned char>& data, size_t& size){
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 调整 data 大小并读取文件内容
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();

    return true;
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

std::string printDims(const nvinfer1::Dims& dims){
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < dims.nbDims; i++){
        oss << dims.d[i];
        if (i != dims.nbDims - 1) {
            oss << ", ";
        }
    }
    oss << " ]";
    return oss.str();
}

std::string printTensor(float* tensor, int size){
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < size; i++){
        oss << std::fixed << std::setprecision(4) << tensor[i];
        if (i != size - 1){
            oss << ", ";
        }
    }
    oss << " ]";
    return oss.str();
}

std::string getEnginePath(const std::string& onnxPath){
    std::filesystem::path onnx_file(onnxPath);
    std::filesystem::path dir = onnx_file.parent_path();
    std::string stem = onnx_file.stem().string(); // 文件名（不含扩展名）

    // 创建 engine 目录
    std::filesystem::path engine_dir = dir / "engine";
    if (!std::filesystem::exists(engine_dir)){
        std::filesystem::create_directory(engine_dir);
    }

    // 构建 engine 文件路径
    std::filesystem::path engine_file = engine_dir / (stem + ".engine");
    return engine_file.string();
}
