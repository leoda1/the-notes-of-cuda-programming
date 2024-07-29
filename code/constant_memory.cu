/******************************************************************
 * Author      : Da Liu
 * Date        : 2024-07-29
 * File Name   : constant_memory.cu
 * Description : 常量内存的使用
 *****************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>
#include "common.cuh"
__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1(void)
{
    printf("constant data c_data = %.2f.\n", c_data);
}

int main(int argc, char** argv)
{
    int device_id = 0;
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    printf("Device %d: %s\n", device_id, device_prop.name);

    float h_data = 3.3f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

    dim3 block(1);
    dim3 grid(1);
    kernel_1<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
    printf("constant data h_data = %.2f.\n", h_data);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}