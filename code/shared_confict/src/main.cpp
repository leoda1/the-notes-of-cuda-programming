#include <iostream>
#include "cuda_runtime.h"
#include <stdio.h>

#include "../inc/timer.hpp"
#include "../inc/utils.hpp"
#include "../inc/matmul.hpp"

int main()
{
    Timer timer;
    int width = 1 << 12;
    int size = width * width;
    int low = 0, high = 1;

    int block_size = 16;
    bool stateMem = true;
    char str[100];
    int seed = 0;

    float* h_M = (float*)malloc(size*sizeof(float));
    float* h_N = (float*)malloc(size*sizeof(float));
    float* h_P = (float*)malloc(size*sizeof(float));
    float* d_P = (float*)malloc(size*sizeof(float));

    //initialize matrices
    seed = 1;
    initMatrix(h_M, size, low, high, seed);
    seed = 2;
    initMatrix(h_N, size, low, high, seed);
    LOG("Input matrix size: %d x %d", width, width);

    /* GPU warmup*/
    timer.start_gpu();
    Matmul_device(h_M, h_N, h_P, width, block_size);
    timer.stop_gpu();
    timer.duration_gpu("GPU warmup(use gpu):");















}