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

__global__ void Matmul_shared_static_kernel(float* M_device, float* N_device, float* P_device, int width, bool bank_conflict)
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
    if(bank_conflict){
        for (int m = 0; m < width / BLOCKSIZE; m++){
            M_device_shared[tx][ty] = M_device[ix * width + (m * BLOCKSIZE + ty)];     
            N_device_shared[tx][ty] = N_device[(m * BLOCKSIZE + tx) * width + iy];
            __syncthreads();
            for (int k = 0; k < BLOCKSIZE; k++){
                P_element += M_device_shared[tx][k] * N_device_shared[k][ty];
            }
            __syncthreads();
        }
        P_device[idx] = P_element;
    }else{
        for (int m = 0; m < width / BLOCKSIZE; m++){
            M_device_shared[ty][tx] = M_device[iy * width + (m * BLOCKSIZE + tx)];       
            N_device_shared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + ix];
            __syncthreads();
            for (int k = 0; k < BLOCKSIZE; k++){
                P_element += M_device_shared[ty][k] * N_device_shared[k][tx];
            }
            __syncthreads();
        }
        P_device[idx] = P_element;
    }
}

__global__ void Matmul_shared_dynamic_kernel(float* M_device, float* N_device, float* P_device, int width, int blocksize, bool bank_conflict)
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
    if (bank_conflict){
        for (int i = 0; i < width / blocksize; i++){
            device_shared[ty + tx * blocksize] = M_device[ix * width + (i * blocksize + ty)];
            device_shared[ty + tx * blocksize + stride] = N_device[(i * blocksize + tx) * width + iy];
            __syncthreads();
            for (int j = 0; j < blocksize; j++){
                P_element += device_shared[ty + j * blocksize] * device_shared[ty + tx * blocksize + j];
            }
            __syncthreads();
        }
        P_device[idx] = P_element;

    }else{
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
    
}   
/*
    普通的矩阵乘法
*/
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
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_M));
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile
*/
void Matmul_device_shared(float* h_M, float* h_N, float* h_P, int width, int blocksize, bool staticMem, bool bank_conflict)
{
    int size = width * width * sizeof(float);
    long int sMemSize = blocksize * blocksize * sizeof(float) * 2;
    //allocate memory for M, N, and P
    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, size));
    CUDA_CHECK(cudaMalloc((void**)&d_N, size));
    CUDA_CHECK(cudaMalloc((void**)&d_P, size));
    //copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));
    //launch the kernel
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(width / blocksize, width / blocksize);
    if (staticMem){
        Matmul_shared_static_kernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width, bank_conflict);
    }else{
        Matmul_shared_dynamic_kernel<<<dimGrid, dimBlock, sMemSize, nullptr>>>(d_M, d_N, d_P, width, blocksize, bank_conflict);
    }
    //copy data from device to host
    CUDA_CHECK(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));
    //synchronize the device
    CUDA_CHECK(cudaDeviceSynchronize());
    //check kernel errors
    LAST_KERNEL_CHECK();
    //free device memory
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_M));
}
