/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-28
 * File: static_shared_memory.cu
 * Description: 静态gpu共享内存
 ***************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include "common.cuh"

__global__ void kernel_1(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    
    __shared__ float s_array[32];
    if ( n < N)
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
    std::cout << "Device " << device_id << ": " << device_prop.name << std::endl;

    int nElem = 64;
    int Bytes = nElem * sizeof(float);

    float* h_A = nullptr;
    h_A = (float*)malloc(Bytes);
    for (int i = 0; i < nElem; i++) {
        h_A[i] = float(i);
    }

    float* d_A = nullptr;
    ErrorCheck(cudaMalloc(&d_A, Bytes), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_A, h_A, Bytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block(32);
    dim3 grid(2);

    kernel_1<<<grid, block>>>(d_A, nElem);
    ErrorCheck(cudaFree(d_A), __FILE__, __LINE__);
    free(h_A);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}