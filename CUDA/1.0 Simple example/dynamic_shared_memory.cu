/******************************************************************
 * Author      : Da Liu
 * Date        : 2024-07-28
 * File Name   : dynamic_shared_memory.cu
 * Description : 动态共享内存使用
 *****************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

extern __shared__ float s_array[];                         // 声明动态共享内存变量, 这里必须是数组


__global__ void kernel(float* d_A, const int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;

    if (n < N)
        s_array[tid] = d_A[n]; 
    __syncthreads();
    if (tid == 0)
        for (int i = 0; i < 32; i++)
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
}


int main(int argc, char** argv)
{
    int device_id = 0;
    cudaDeviceProp device_prop;
    ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);
    std::cout << "Device name: " << device_prop.name << std::endl;

    int nElem = 64;
    int nBytes = nElem * sizeof(float);
    float* p_A = nullptr;
    p_A = (float*)malloc(nBytes);
    for (int i = 0; i < nElem; i++)
        p_A[i] = float(i);

    float* d_A = nullptr;
    ErrorCheck(cudaMalloc(&d_A, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_A, p_A, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block(32);
    dim3 grid(2);

    kernel<<<grid, block>>>(d_A, nElem);
    ErrorCheck(cudaFree(d_A), __FILE__, __LINE__);
    free(p_A);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;

}