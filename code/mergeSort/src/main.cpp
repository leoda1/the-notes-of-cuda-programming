#include <assert.h> //断言
#include <stdio.h>  //标准输入输出
#include <stdlib.h> //提供内存分配、程序控制功能
#include <cuda_runtime.h> //CUDA运行时库
#include "../inc/helper_functions.h"
#include "../inc/helper_cuda.h"
#include "../inc/mergeSort_common.h"

int main(int argc, char **argv) {
    uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;                        // host memory
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;  // device memory
    StopWatchInterface *hTimer = NULL;                                      // 计时器对象指针

    const uint N = 4 * 1048576;             // matrix size
    const uint DIR = 1;                     // 1 for ascending, -1 for descending
    const uint numValues = 65536;           // number of values to sort

    printf("%s Starting...\n\n", argv[0]);  //打印程序启动信息 

    int dev = findCudaDevice(argc, (const char **)argv);                    // find the device
    if (dev == -1) {
        return EXIT_FAILURE;                                                // no CUDA device found
    }

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);                                                // 创建计时器
    /**************malloc memory***************/
    h_SrcKey = (uint *)malloc(N * sizeof(uint));
    h_SrcVal = (uint *)malloc(N * sizeof(uint));
    h_DstKey = (uint *)malloc(N * sizeof(uint));
    h_DstVal = (uint *)malloc(N * sizeof(uint));

    srand(2009);                                                             // 设置随机数种子  
    for (uint i = 0; i < N; i++) {
        h_SrcKey[i] = rand() % numValues;                                    // 随机生成key
    }

    fillValues(h_SrcVal, N);                                                  // 填充随机值 

    printf("Allocating and initializing CUDA arrays...\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
    checkCudaErrors(
        cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Initializing GPU merge sort...\n");
    initMergeSort();                                                          // 初始化GPU merge sort

    printf("Running GPU merge sort...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);                                                   // 重置计时器    
    sdkStartTimer(&hTimer);                                                   // 开始计时器                 
    mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, DIR);//调用归并排序函数
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));  //打印耗时

    printf("Reading back GPU merge sort results...\n");
    checkCudaErrors(
        cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf("Inspecting the results...\n");
    uint keysFlag = validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);

    uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

    printf("Shutting down...\n");
    closeMergeSort();
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_SrcVal));
    checkCudaErrors(cudaFree(d_SrcKey));
    checkCudaErrors(cudaFree(d_BufVal));
    checkCudaErrors(cudaFree(d_BufKey));
    checkCudaErrors(cudaFree(d_DstVal));
    checkCudaErrors(cudaFree(d_DstKey));
    free(h_DstVal);
    free(h_DstKey);
    free(h_SrcVal);
    free(h_SrcKey);

    exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}