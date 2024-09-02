#include <stdio.h>
#include <iostream>
#include "cuda_runtime_api.h"

#include "../inc/utils.hpp"

__global__ void Matmul_device_kernel(float* d_M, float* d_N, float* d_P, int width){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * width + ix;

    float P_element = 0;
    for ( int i = 0; i < width; i++ ) {
        float M_element = d_M[iy * width + i];
        float N_element = d_N[i * width + ix];
        P_element += M_element * N_element;
    }
    d_P[idx] = P_element;
}

void Matmul_device(float* h_M, float* h_N, float* h_P, int width, int block_size)
{
    int size = width * width * sizeof(float);
    // Allocate device memory for matrices
    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, size));
    CUDA_CHECK(cudaMalloc((void**)&d_N, size));
    CUDA_CHECK(cudaMalloc((void**)&d_P, size));
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));
    // Launch the kernel
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(width / block_size, width / block_size);
    Matmul_device_kernel<<<dim_grid, dim_block>>>(d_M, d_N, d_P, width);
    //copy data from device to host
    CUDA_CHECK(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));
    //synchronize threads
    CUDA_CHECK(cudaDeviceSynchronize());
    //check kernel errors
    LAST_KERNEL_CHECK();
    // Free device memory
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
}


void Matmul_device_shared(float* h_M, float* h_N, float* h_P, int width, int block_size)
{

}