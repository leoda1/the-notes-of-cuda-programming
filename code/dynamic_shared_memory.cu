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
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    std::cout << "Device name: " << device_prop.name << std::endl;

    int nElem = 64;
    int nBytes = nElem * sizeof(float);
    float* p_A = nullptr;
    p_A = (float*)malloc(nBytes);
    for (int i = 0; i < nElem; i++)
        p_A[i] = float(i);

    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, nBytes));
    CUDA_CHECK(cudaMemcpy(d_A, p_A, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);

    kernel<<<grid, block>>>(d_A, nElem);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaDeviceReset());
    free(p_A);

    return 0;

}