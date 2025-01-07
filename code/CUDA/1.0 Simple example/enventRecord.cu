#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "common.cuh"

#define NUM 10

__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void addFromGPU(float* A, float* B, float* C, const int n)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;
    if (idx > n) return;
    C[idx] = add(A[idx], B[idx]);
}

void initialData(float* addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xff) / 10.f;
    }
    return;
}

int main()
{
    setGPU();                                               //设置GPU设备
    int iElemCount = 4096;                                   //设置元素数量
    size_t stByteCount = iElemCount * sizeof(float);        //设置字节数量
    //allocate host memory
    float* fpHost_A, * fpHost_B, * fpHost_C;
    fpHost_A = (float*)malloc(stByteCount);
    fpHost_B = (float*)malloc(stByteCount);
    fpHost_C = (float*)malloc(stByteCount);

    if (fpHost_A !=NULL && fpHost_B !=NULL && fpHost_C !=NULL)
    {
        memset(fpHost_A, 0, stByteCount);
        memset(fpHost_B, 0, stByteCount);
        memset(fpHost_C, 0, stByteCount);                   //初始化数据
    }
    else
    {
        printf("allocate host memory failed!\n");
    //    cout << "allocate host memory failed!" << endl;
        exit(-1);
    }
    //allocate device memory
    float* fpDev_A, * fpDev_B, * fpDev_C;
    cudaMalloc((float**)&fpDev_A, stByteCount);
    cudaMalloc((float**)&fpDev_B, stByteCount);
    cudaMalloc((float**)&fpDev_C, stByteCount);
    if(fpDev_A !=NULL && fpDev_B !=NULL && fpDev_C !=NULL)
    {
        cudaMemset(fpDev_A, 0, stByteCount);
        cudaMemset(fpDev_B, 0, stByteCount);
        cudaMemset(fpDev_C, 0, stByteCount);                   //初始化数据
    }
    else
    {
        printf("allocate device memory failed!\n");
    //    cout << "allocate device memory failed!" << endl;
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }
    // initialize data in host memory
    srand(666);
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    // copy data from host to device
    cudaMemcpy(fpDev_A, fpHost_A, stByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDev_B, fpHost_B, stByteCount, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / block.x);
    
    float t_sum = 0;
    for (int i = 0; i < NUM; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        addFromGPU <<< grid, block >>> (fpDev_A, fpDev_B, fpDev_C, iElemCount);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        if (i > 0)
        {
            t_sum += time;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    const float t_avg = t_sum / NUM;

    printf("average t: %g ms.\n", t_avg);
    // copy data from device to host
    cudaMemcpy(fpHost_C, fpDev_C, stByteCount, cudaMemcpyDeviceToHost);
    // free memory
    cudaFree(fpDev_A);
    cudaFree(fpDev_B);
    cudaFree(fpDev_C);
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    return 0;

}