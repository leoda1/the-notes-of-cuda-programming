#include "cuda_runtime.h"
#include "cuda.h"
#include "../inc/utils.hpp"
#include "stdio.h"

__global__ void ReduceNeighboredWithDivergence(float* d_idata, float* d_odata, int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float* idata = d_idata + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}

__global__ void ReduceNeighboredWithoutDivergence(float* d_idata, float* d_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float* idata = d_idata + blockIdx.x * blockDim.x;    
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
            idata[index] += idata[index + stride];
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}

void ReduceOnGPUWithDivergence(float* h_idata, float* h_odata, int size, int block_size)
{
    int ibytes = size * sizeof(float);
    int obytes = size / block_size * sizeof(float);
    memset(h_odata, 0, obytes);
    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
    dim3 block(block_size);
    dim3 grid(size / block_size);
    ReduceNeighboredWithDivergence<<<grid, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK();


    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
}
void ReduceOnGPUWithoutDivergence(float* h_idata, float* h_odata, int size, int block_size)
{

    int ibytes = size * sizeof(float);
    int obytes = size / block_size * sizeof(float);
    memset(h_odata, 0, obytes);
    float* d_idata = nullptr;    
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));    
    dim3 block(block_size);
    dim3 grid(size / block_size);
    ReduceNeighboredWithoutDivergence<<<grid, block>>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
}