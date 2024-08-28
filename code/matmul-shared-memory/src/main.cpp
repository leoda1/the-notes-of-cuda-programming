#include <stdio.h>
#include <cuda_runtime.h>

#include "timer.hpp"
#include "utils.hpp"



int seed;
int main(){
    Timer timer;

    int width = 1 << 12, low = 0, high = 1;
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
    MatmulOndevice();
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu warmup");


    return 0;
}