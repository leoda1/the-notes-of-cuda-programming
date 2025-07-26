#include <cuda.h>
#include <iostream>

#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v2 (float * g_idata, float * g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + tid;
    sdata[tid] = g_idata[idx];
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] +=  sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float *)malloc(N * sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N * sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *output_host = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));
    
    dim3 grid(N / BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v2<<<grid, block>>>(input_device, output_device);
    
    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final = 0.0;
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        if (i < 10) {
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