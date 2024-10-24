#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <ostream>
#include <string>
#include "NvInfer.h"
#include <stdarg.h>
#include <vector>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

#define LOG(...)                     __log_info(Level::INFO, __VA_ARGS__)
#define LOGV(...)                    __log_info(Level::VERB, __VA_ARGS__)
#define LOGE(...)                    __log_info(Level::ERROR, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

enum struct Level {
    ERROR,
    INFO,
    VERB
};

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(Level level, const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);
    int n = 0;

    if (level == Level::INFO) {
        n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]:");
    } else if (level == Level::VERB) {
        n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]:" CLEAR);
    } else {
        n += snprintf(msg + n, sizeof(msg) - n, RED    "[error]:");
    }
    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

bool fileExists(const std::string& fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::vector<unsigned char> loadFile(const std::string &path);
std::string printDims(const nvinfer1::Dims& dims);
std::string printTensor(float* tensor, int size);
std::string getEnginePath(const std::string& onnxPath);

#endif //__UTILS_HPP__
