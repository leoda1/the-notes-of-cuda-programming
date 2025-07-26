#include <cuda.h>
#include <iostream>
#include <time.h>


#define N 32*1024*1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <unsigned int blockSize>
__device__ __forceinline__ float warpreduce(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v7 (float * g_idata, float * g_odata, unsigned int n) {
    float sum = 0;
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + tid;
    
    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sum += g_idata[idx + iter*blockSize];
    }
    
    // do reduction in shared memory
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneID = threadIdx.x % WARP_SIZE;
    const int warpID = threadIdx.x / WARP_SIZE;

    sum = warpreduce<blockSize>(sum);

    if(laneID == 0 )warpLevelSums[warpID] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneID] : 0;
    if (warpID == 0) sum = warpreduce<blockSize / WARP_SIZE>(sum); 
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sum;
}

int main() {
    float *input_host = (float *)malloc(N * sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N * sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    const int block_num = 1024;
    float *output_host = (float*)malloc(block_num * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, block_num * sizeof(float));
    
    dim3 grid(block_num, 1);
    dim3 block(BLOCK_SIZE, 1);
    const int NUM_PER_BLOCK = N / block_num;
    const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
    reduce_v7<BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(input_device, output_device, N);
    
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