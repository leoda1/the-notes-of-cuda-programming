#include <stdio.h>
#include <cuda_runtime.h>

#include "timer.hpp"
#include "utils.hpp"
#include "matmul.hpp"



int seed;
int main(){
    Timer timer;

    int width = 1<<12, low = 0, high = 1;
    int size = width * width;
    int block_size = 16;
    bool statMem = true;
    char str[100];

    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);

    LOG("Input size is:%d x %d", width, width);
    /*GPU warmup*/
    timer.start_gpu();
    Matmul_device(h_matM, h_matN, h_matP, width, block_size);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu warmup");
    /*kernel function  <<<24, 32>>>*/
    timer.start_gpu();
    Matmul_device(h_matM, h_matN, d_matP, width, block_size);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu without shared memory)<<<%d, %d>>>",width / block_size, block_size);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);
    return 0;
}
/******************************************************************
(joker) PS C:\Users\22681\Desktop\project\cudalearn\notes\code\matmul-shared-memory> ./build/Debug/matmul.exe
Input size is:768 x 768
matmul in gpu warmup                                         uses 3.410112 ms
matmul in gpu without shared memory)<<<24, 32>>>             uses 3.022272 ms
(joker) PS C:\Users\22681\Desktop\project\cudalearn\notes\code\matmul-shared-memory> ./build/Debug/matmul.exe
Input size is:4096 x 4096
matmul in gpu warmup                                         uses 187.220901 ms
matmul in gpu without shared memory)<<<256, 16>>>            uses 184.976639 ms
 *****************************************************************/