#include <cuda.h>
#include <iostream>

#define N 32*1024*1024
#define BLOCK_SIZE 256

template <unsigned int blockSize>
__device__ void warpreduce(volatile float* sdata, unsigned int tid) {
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5 (float * g_idata, float * g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;
    sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x];
    __syncthreads();

    // do reduction in shared memory
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }

    if (tid < 32) warpreduce<blockSize>(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float *)malloc(N * sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N * sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    float *output_host = (float*)malloc(block_num * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, block_num * sizeof(float));
    
    dim3 grid(block_num, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v5<BLOCK_SIZE><<<grid, block>>>(input_device, output_device);
    
    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final = 0.0;
    for (int i = 0; i < block_num; i++) {
    if (i < 10){
            std::cout << "Block " << i << " result: " << output_host[i] << std::endl;
        }
        final += output_host[i];  // Sum up the block results
    }
    std::cout << "Final result after reduction: " << final << std::endl;
    float expected = 2.0f * N;
    std::cout << "Expected final result: " << expected << std::endl;

    // Free memory
    cudaFree(input_device);
    cudaFree(output_device);
    free(input_host);
    free(output_host);
    return 0;
}