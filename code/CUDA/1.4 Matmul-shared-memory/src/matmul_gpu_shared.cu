#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <stdio.h>
#include "../inc/utils.hpp"
#define BLOCKSIZE 32

__global__ void Matmul_shared_static_kernel(float* M_device, float* N_device, float* P_device, int width)
{

    __shared__ float M_device_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_device_shared[BLOCKSIZE][BLOCKSIZE];
    //index of the current element in the matrix
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * width + ix;
    //index of tiles in the matrix
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float P_element = 0;
    for (int i = 0; i < width / BLOCKSIZE; i++){
/*      iy * width is the row index,  i * BLOCKSIZE + tx is the column index*/
        M_device_shared[ty][tx] = M_device[iy * width + (i * BLOCKSIZE + tx)];
/*      (i * BLOCKSIZE + ty) * width is the row index,  ix is the column index*/       
        N_device_shared[ty][tx] = N_device[(i * BLOCKSIZE + ty) * width + ix];
        __syncthreads();
        for (int j = 0; j < BLOCKSIZE; j++){
            P_element += M_device_shared[ty][j] * N_device_shared[j][tx];
        }
        __syncthreads();
    }
    P_device[idx] = P_element;
}


__global__ void Matmul_shared_dynamic_kernel(float* M_device, float* N_device, float* P_device, int width, int blocksize)
{
    /*
        dynamic shared memory allocation
        Notice: 1.the address of shared memory is the same, and two pointers are needed to point to different locations
                2.When declaring a dynamic shared variable, you need to add "extern" and it needs to be one-dimensional.
    */
    extern __shared__ float device_shared[];
    int stride = blocksize * blocksize;

    //index of the current element in the matrix
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * width + ix;
    //index of tiles in the matrix
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float P_element = 0;
    for (int i = 0; i < width / blocksize; i++){
        device_shared[tx + ty * blocksize] = M_device[iy * width + (i * blocksize + tx)];
        device_shared[tx + ty * blocksize + stride] = N_device[(i * blocksize + ty) * width + ix];
        __syncthreads();
        for (int j = 0; j < blocksize; j++){
            P_element += device_shared[tx + j * blocksize] * device_shared[tx + ty * blocksize + j];
        }
        __syncthreads();
    }
    P_device[idx] = P_element;
}   

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile
*/
void Matmul_shared_memory(float* M_host, float* N_host, float* P_host, int width, int blocksize, bool statMem)
{
    int size = width * width * sizeof(float);
    long int sMemSize = blocksize * blocksize * sizeof(float) * 2;
    //allocate memory for M, N, and P
    float*M_device, *N_device, *P_device;
    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));
    //copy data from host to device
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));
    //launch the kernel
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(width / blocksize, width / blocksize);
    if (statMem){
        Matmul_shared_static_kernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    }else{
        Matmul_shared_dynamic_kernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width, blocksize);
    }
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