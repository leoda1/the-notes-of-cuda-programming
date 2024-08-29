#include "cuda_runtime_api.h"
#include <stdio.h>
#include <iostream>
#include "../inc/utils.hpp"

/*乘法函数*/
__global__ void Matmul_kernel(float* M_device, float* N_device, float* P_device, int width)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * width + ix;

    float P_element = 0;
    for (int k = 0; k < width; k ++){
        float M_element = M_device[iy * width + k];
        float N_element = N_device[k * width + ix];
        P_element += M_element * N_element;
    }
    P_device[idx] = P_element;
}

void Matmul_device(float* M_host, float* N_host, float* P_host, int width, int blocksize)
{
    int size = width * width * sizeof(float);//the number of bytes occupied by the square width
    //allocate memory for M, N, and P
    float *M_device, *N_device, *P_device;
    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));
    //copy data from host to device
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));
    //launch the kernel
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(width / blocksize, width / blocksize);
    Matmul_kernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    //copy data from device to host
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    //synchronize the device
    CUDA_CHECK(cudaDeviceSynchronize());
    //check kernel errors
    LAST_KERNEL_CHECK();
    //free device memory
    CUDA_CHECK(cudaFree(M_device));
    CUDA_CHECK(cudaFree(N_device));
    CUDA_CHECK(cudaFree(P_device));
}